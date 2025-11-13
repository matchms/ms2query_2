import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import nmslib
import numpy as np
from scipy import sparse as sp
from ms2query.metrics import tanimoto_l1_query_vs_block


# ======================
# Utilities
# ======================

def tuples_to_csr(
    items: Sequence[Tuple[np.ndarray, np.ndarray]], dim: int
) -> sp.csr_matrix:
    """
    Build a CSR (N, dim) from sequence of (indices_uint32, values_float32).
    Coalesces duplicates per row. Enforces float32 values.
    """
    N = len(items)
    indptr = np.empty(N + 1, dtype=np.int64)
    indptr[0] = 0
    indices_list = []
    data_list = []
    for i, (idxs, vals) in enumerate(items):
        idxs = np.asarray(idxs, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float32)
        if idxs.size and idxs.max() >= dim:
            raise ValueError(f"Row {i}: index {idxs.max()} >= dim {dim}")
        order = np.argsort(idxs, kind="mergesort")
        idxs = idxs[order]
        vals = vals[order]
        if idxs.size > 1:
            dup = idxs[1:] == idxs[:-1]
            if dup.any():
                # compress duplicates
                uniq, start = np.unique(idxs, return_index=True)
                vals = np.add.reduceat(vals, start)
                idxs = uniq
        indices_list.append(idxs)
        data_list.append(vals)
        indptr[i + 1] = indptr[i] + idxs.size
    if N:
        indices = np.concatenate(indices_list) if indices_list else np.empty(0, np.int32)
        data = np.concatenate(data_list) if data_list else np.empty(0, np.float32)
    else:
        indices = np.empty(0, np.int32)
        data = np.empty(0, np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(N, dim), dtype=np.float32)

def csr_row_from_tuple(item: Tuple[np.ndarray, np.ndarray], dim: int) -> sp.csr_matrix:
    idxs = np.asarray(item[0], dtype=np.int32)
    vals = np.asarray(item[1], dtype=np.float32)
    if idxs.size and idxs.max() >= dim:
        raise ValueError(f"Query index {idxs.max()} >= dim {dim}")
    order = np.argsort(idxs, kind="mergesort")
    idxs = idxs[order]
    vals = vals[order]
    if idxs.size > 1 and (idxs[1:] == idxs[:-1]).any():
        uniq, start = np.unique(idxs, return_index=True)
        vals = np.add.reduceat(vals, start)
        idxs = uniq
    indptr = np.array([0, idxs.size], dtype=np.int64)
    return sp.csr_matrix((vals, idxs, indptr), shape=(1, dim), dtype=np.float32)

def l1_norms_csr(X: sp.csr_matrix) -> np.ndarray:
    # float64 for safety with very large counts
    return np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)


# ======================
# Base class
# ======================

@dataclass
class _BaseANN:
    dim: int
    space: str
    _meta: dict = None

    def save_index(self, path_prefix: str) -> None:
        raise NotImplementedError

    def load_index(self, path_prefix: str) -> None:
        raise NotImplementedError

    def build_index(self, *args, **kwargs):
        raise NotImplementedError

    def build_index_from_sqlite(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError


# ======================
# ANN-1: MS2Deepscore embeddings (dense cosine, nmslib)
# ======================

class EmbeddingIndex(_BaseANN):
    """
    Dense cosine ANN using nmslib (HNSW).
    - build_index(vectors, spec_ids, ...)
    - build_index_from_sqlite(sqlite_conn or SpectralDatabase, embeddings_table='embeddings', ...)
    - query(vector, k)
    - save_index / load_index
    """

    def __init__(self, dim: int = 500):
        super().__init__(dim=dim, space="cosinesimil")
        self._index = None
        self._comp_ids: Optional[np.ndarray] = None

    def build_index(
        self,
        vectors: np.ndarray,
        spec_ids: Iterable[str],
        *,
        M: int = 16,
        ef_construction: int = 200,
        post_init_ef: int = 200,
    ) -> None:
        """Build index from dense vectors and spec_ids.
        
        Parameters
        ----------
        vectors : np.ndarray
            2D array of shape (N, dim) with float32 vectors.
        spec_ids : Iterable[str]
            Iterable of spec_id strings of length N.
        M : int
            HNSW M parameter (connectivity)
        ef_construction : int
            HNSW efConstruction parameter
        post_init_ef : int
            HNSW query-time ef parameter
        """
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (N, {self.dim}), got {X.shape}")
        ids = np.asarray(list(spec_ids), dtype=object)
        if ids.shape[0] != X.shape[0]:
            raise ValueError("spec_ids length must match number of vectors.")

        index = nmslib.init(method='hnsw', space=self.space, data_type=nmslib.DataType.DENSE_VECTOR)
        index.addDataPointBatch(X)              # can be called multiple times before createIndex
        index.createIndex({'M': M, 'efConstruction': ef_construction}, print_progress=False)
        index.setQueryTimeParams({'ef': post_init_ef})

        self._index = index
        self._ids = ids
        self._meta = {
            "type": "ANNMS2DeepIndex",
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
        }

    def _count_rows(self, cursor: sqlite3.Cursor, table: str, where_clause: str) -> int:
        sql = f"SELECT COUNT(1) FROM {table} {where_clause};"
        cursor.execute(sql)
        (n,) = cursor.fetchone()
        return int(n)

    def build_index_from_sqlite(
        self,
        db: sqlite3.Connection,
        *,
        embeddings_table: str = "embeddings",
        where_sql: Optional[str] = None,
        batch_rows: int = 100_000,
        M: int = 16,
        ef_construction: int = 200,
        post_init_ef: int = 200,
    ) -> int:
        """
        Streams embeddings from SQLite and constructs an HNSW index in-place.
        Indexing is performed with a SINGLE addDataPointBatch call to avoid
        backend-specific issues when adding multiple batches before createIndex.

        Parameters
        ----------
        db : sqlite3.Connection
            Database connection (or SpectralDatabase.connection)
        embeddings_table : str
            Name of the table containing embeddings
        where_sql : Optional[str]
            Optional filter clause (e.g., "d=500" or "WHERE d=500")
        batch_rows : int
            Number of rows to process before adding to index (memory tuning)
        M : int
            HNSW M parameter (connectivity)
        ef_construction : int
            HNSW efConstruction parameter
        post_init_ef : int
            HNSW query-time ef parameter

        Returns
        -------
        int
            Number of vectors indexed
        """
        cur = db.cursor()

        # Detect and validate dimension
        d = self._detect_dimension(cur, embeddings_table)
        if self.dim != d:
            self.dim = d  # adopt DB dimension

        # Build WHERE clause & count rows up-front
        where_clause = self._build_where_clause(where_sql)
        total = self._count_rows(cur, embeddings_table, where_clause)
        if total == 0:
            raise ValueError(f"No embeddings loaded from {embeddings_table}.")

        # Pre-allocate one dense (N, d) array and ids array
        X = np.empty((total, d), dtype=np.float32)
        ids = np.empty(total, dtype=object)

        # Stream rows and fill pre-allocated buffers in deterministic order
        sql = f"SELECT spec_id, vec FROM {embeddings_table} {where_clause} ORDER BY spec_id ASC;"
        cur.execute(sql)

        filled = 0
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            for sid, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32, count=d)
                if vec.size != d:
                    raise ValueError(
                        f"Embedding for '{sid}' has {vec.size} dimensions, expected {d}."
                    )
                X[filled] = vec  # copies from the buffer
                ids[filled] = str(sid)
                filled += 1
        cur.close()

        if filled != total:
            # defensive: table changed mid-scan — shrink to what we actually loaded
            X = X[:filled]
            ids = ids[:filled]
            total = filled
            if total == 0:
                raise ValueError(f"No embeddings loaded from {embeddings_table}.")

        # Build the HNSW index with a SINGLE batch add
        index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
        index.addDataPointBatch(X)
        index.createIndex({'M': M, 'efConstruction': ef_construction}, print_progress=False)
        index.setQueryTimeParams({'ef': post_init_ef})

        # Commit
        self._index = index
        self._ids = ids
        self._meta = {
            "type": "ANNMS2DeepIndex",
            "built_from_sqlite": True,
            "embeddings_table": embeddings_table,
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
        }
        return int(total)

    def _detect_dimension(self, cursor: sqlite3.Cursor, table: str) -> int:
        """Detect and validate embedding dimension from table."""
        sql = f"SELECT DISTINCT d FROM {table} ORDER BY d;"
        dims = [int(row[0]) for row in cursor.execute(sql)]
        
        if not dims:
            raise ValueError(f"No rows found in table '{table}'.")
        if len(dims) > 1:
            raise ValueError(
                f"Mixed dimensions in table '{table}': {dims}. "
                f"All embeddings must have the same dimension."
            )
        
        return dims[0]

    def _build_where_clause(self, where_sql: Optional[str]) -> str:
        """Build WHERE clause from user input, handling various formats."""
        if not where_sql:
            return ""
        
        where_sql = where_sql.strip()
        if where_sql.upper().startswith("WHERE"):
            return where_sql
        else:
            return f"WHERE {where_sql}"

    # ---------- querying ----------
    def query(
            self,
            vector: np.ndarray,
            k: int = 10,
            ef: Optional[int] = None,
            ) -> List[Tuple[str, float]]:
        """Query the index with a single vector.
    
        Parameters
        ----------
        vector : np.ndarray
            1D array of shape (dim,) with float32 vector.
        k : int
            Number of nearest neighbors to return.
        ef : Optional[int]
            nmslib ef parameter (higher = better recall / slower).
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != self.dim:
            raise ValueError(f"Query vector must have dim={self.dim}")

        if ef is not None:
            self._index.setQueryTimeParams({'ef': int(ef)})

        res = self._index.knnQueryBatch(v, k=k)   # returns [(idxs, dists)]
        idxs, dists = res[0]
        idxs = np.asarray(idxs, dtype=np.int64)
        dists = np.asarray(dists, dtype=np.float32)
        sims = 1.0 - dists  # cosinesimil → distance = 1 - cosine
        return [(str(self._ids[i]), float(sims[j])) for j, i in enumerate(idxs)]

    # ---------- persistence ----------
    def save_index(self, path_prefix: str) -> None:
        """Save index to files with given prefix.
        """
        if self._index is None or self._ids is None:
            raise RuntimeError("Index not built or loaded.")
        meta_path = f"{path_prefix}.meta.json"
        ids_path = f"{path_prefix}.ids.npy"
        hnsw_path = str(path_prefix)  #f"{path_prefix}.nmslib"
        self._index.saveIndex(hnsw_path, save_data=True)
        np.save(ids_path, self._ids)
        meta = dict(self._meta or {})
        meta.update({"dim": self.dim, "space": self.space})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load_index(self, path_prefix: str) -> None:
        """Load index from files with given prefix.
        """
        meta_path = f"{path_prefix}.meta.json"
        ids_path = f"{path_prefix}.ids.npy"
        hnsw_path = str(path_prefix)  #f"{path_prefix}.nmslib"
        if not (os.path.exists(meta_path) and os.path.exists(ids_path) and os.path.exists(hnsw_path)):
            raise FileNotFoundError("Missing files for EmbeddingIndex.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = int(meta["dim"])
        self.space = str(meta["space"])
        self._meta = meta
        self._ids = np.load(ids_path, allow_pickle=True)
        self._index = nmslib.init(method='hnsw', space=self.space, data_type=nmslib.DataType.DENSE_VECTOR)
        self._index.loadIndex(hnsw_path, load_data=True)


# ======================
# ANN-2: Sparse fingerprints (cosine ANN + exact L1/Jaccard Tanimoto re-rank)
# ======================

class FingerprintSparseIndex(_BaseANN):
    """
    Sparse fingerprints (non-negative counts/weights), million-scale ready.
    - ANN: nmslib HNSW over sparse cosine ('cosinesimil_sparse').
    - Re-rank: exact generalized Tanimoto (L1/Jaccard) on top-k*mult candidates,
               using CSR two-pointer merge (numba), no densification.
    """

    def __init__(self, dim: int = 4096):
        super().__init__(dim=dim, space="cosinesimil_sparse")
        self._index = None
        self._comp_ids: Optional[np.ndarray] = None
        self._csr: Optional[sp.csr_matrix] = None     # DB in CSR
        self._l1: Optional[np.ndarray] = None         # L1 norms (float64)

    def build_index(
        self,
        data: sp.csr_matrix | Sequence[Tuple[np.ndarray, np.ndarray]],
        comp_ids: Iterable[int],
        *,
        dim: Optional[int] = None,
        M: int = 32,
        ef_construction: int = 300,
        post_init_ef: int = 300,
        keep_csr_for_rerank: bool = True,
        compute_l1_for_rerank: bool = True,
    ) -> None:
        if isinstance(data, sp.csr_matrix):
            csr = data.astype(np.float32, copy=False)
            D = csr.shape[1]
        else:
            D = int(dim if dim is not None else self.dim)
            if D is None:
                raise ValueError("dim must be provided when building from tuples.")
            csr = tuples_to_csr(data, dim=D)

        if (csr.data < 0).any():
            raise ValueError("Fingerprints must be non-negative for Tanimoto.")

        self.dim = D
        comp_ids = np.asarray(list(comp_ids))
        if comp_ids.shape[0] != csr.shape[0]:
            raise ValueError("comp_ids length must match number of rows.")

        # ANN over sparse cosine
        index = nmslib.init(method='hnsw', space=self.space, data_type=nmslib.DataType.SPARSE_VECTOR)
        index.addDataPointBatch(csr)
        index.createIndex({'M': M, 'efConstruction': ef_construction}, print_progress=False)
        index.setQueryTimeParams({'ef': post_init_ef})

        self._index = index
        self._comp_ids = comp_ids
        self._csr = csr if keep_csr_for_rerank else None
        self._l1 = l1_norms_csr(csr) if compute_l1_for_rerank else None
        self._meta = {
            "type": "ANNFingerprintSparseIndex",
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
            "keep_csr_for_rerank": bool(keep_csr_for_rerank),
            "compute_l1_for_rerank": bool(compute_l1_for_rerank),
        }

    def query(
        self,
        query_fp: Tuple[np.ndarray, np.ndarray] | sp.csr_matrix,
        k: int = 10,
        *,
        ef: Optional[int] = None,
        re_rank: bool = True,
        candidate_multiplier: int = 5,
    ) -> List[Tuple[int, float]]:
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")

        # Normalize query to 1×D CSR
        if isinstance(query_fp, sp.csr_matrix):
            q = query_fp
            if q.shape[0] != 1:
                raise ValueError("CSR query must have shape (1, D).")
            if q.shape[1] != self.dim:
                raise ValueError(f"CSR query dim mismatch: got {q.shape[1]}, expected {self.dim}")
        else:
            q = csr_row_from_tuple(query_fp, dim=self.dim)
        if (q.data < 0).any():
            raise ValueError("Query fingerprint must be non-negative for Tanimoto.")
        if q.nnz == 0:
            return []

        if ef is not None:
            self._index.setQueryTimeParams({'ef': int(ef)})

        fetch = max(k, int(k * candidate_multiplier))

        # Use batch API for CSR input
        res = self._index.knnQueryBatch(q, k=fetch)
        idxs, dists = res[0]
        idxs = np.asarray(idxs, dtype=np.int64)
        dists = np.asarray(dists, dtype=np.float32)

        if not re_rank or self._csr is None or self._l1 is None:
            sims = 1.0 - dists  # cosinesimil_sparse: distance = 1 - cosine
            return [(int(self._comp_ids[i]), float(s)) for i, s in zip(idxs[:k], sims[:k])]

        # Exact L1/Jaccard Tanimoto on candidates (no densification)
        Y = self._csr[idxs]
        sum1 = float(q.sum())  # L1 of query
        sumsY = self._l1[idxs]  # L1 of candidates (float64)
        tan = tanimoto_l1_query_vs_block(q, Y, sum1=sum1, sumsY=sumsY)
        order = np.argsort(-tan)
        idxs_sorted = idxs[order][:k]
        scores_sorted = tan[order][:k]
        return [(int(self._comp_ids[i]), float(s)) for i, s in zip(idxs_sorted, scores_sorted)]

    # -------- persistence --------

    def save_index(self, path_prefix: str) -> None:
        if self._index is None or self._comp_ids is None:
            raise RuntimeError("Index not built or loaded.")

        meta_path = f"{path_prefix}.meta.json"
        ids_path = f"{path_prefix}.ids.npy"
        hnsw_path = f"{path_prefix}.nmslib"
        csr_path = f"{path_prefix}.csr.npz"
        l1_path = f"{path_prefix}.l1.npy"

        self._index.saveIndex(hnsw_path)
        np.save(ids_path, self._comp_ids)
        meta = dict(self._meta or {})
        meta.update({"dim": self.dim, "space": self.space})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        if self._csr is not None:
            sp.save_npz(csr_path, self._csr, compressed=True)
        elif os.path.exists(csr_path):
            os.remove(csr_path)

        if self._l1 is not None:
            np.save(l1_path, self._l1)
        elif os.path.exists(l1_path):
            os.remove(l1_path)

    def load_index(self, path_prefix: str) -> None:
        meta_path = f"{path_prefix}.meta.json"
        ids_path = f"{path_prefix}.ids.npy"
        hnsw_path = f"{path_prefix}.nmslib"
        csr_path = f"{path_prefix}.csr.npz"
        l1_path = f"{path_prefix}.l1.npy"

        if not (os.path.exists(meta_path) and os.path.exists(ids_path) and os.path.exists(hnsw_path)):
            raise FileNotFoundError("Missing files for ANNFingerprintSparseIndex.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = int(meta["dim"])
        self.space = str(meta["space"])
        self._meta = meta

        self._comp_ids = np.load(ids_path, allow_pickle=False)
        self._index = nmslib.init(method='hnsw', space=self.space, data_type=nmslib.DataType.SPARSE_VECTOR)
        self._index.loadIndex(hnsw_path, load_data=True)

        self._csr = sp.load_npz(csr_path).astype(np.float32, copy=False) if os.path.exists(csr_path) else None
        self._l1 = np.load(l1_path, allow_pickle=False) if os.path.exists(l1_path) else None
