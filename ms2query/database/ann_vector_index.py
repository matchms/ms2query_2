import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import nmslib
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
from umap import UMAP
from ms2query.metrics import tanimoto_l1_query_vs_block


# ======================
# Utilities
# ======================

def tuples_to_csr(
    items: Sequence[Tuple[np.ndarray, np.ndarray]], dim: int
) -> sp.csr_matrix:
    """
    Build a CSR (N, dim) from sequence of (indices, values).
    Coalesces duplicate indices per row.
    """
    N = len(items)
    indptr = np.empty(N + 1, dtype=np.int64)
    indptr[0] = 0
    indices_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    for i, (idxs, vals) in enumerate(items):
        idxs, vals = _coalesce_sparse_row(idxs, vals, dim, row_id=i)
        indices_list.append(idxs)
        data_list.append(vals)
        indptr[i + 1] = indptr[i] + idxs.size

    indices = np.concatenate(indices_list) if indices_list else np.empty(0, np.int32)
    data = np.concatenate(data_list) if data_list else np.empty(0, np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(N, dim), dtype=np.float32)


def _coalesce_sparse_row(
    idxs: np.ndarray,
    vals: np.ndarray,
    dim: int,
    row_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort indices and sum duplicate values."""
    idxs = np.asarray(idxs, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float32)

    if idxs.size and idxs.max() >= dim:
        ctx = f"Row {row_id}: " if row_id is not None else ""
        raise ValueError(f"{ctx}index {idxs.max()} >= dim {dim}")

    if idxs.size == 0:
        return idxs, vals

    order = np.argsort(idxs, kind="mergesort")
    idxs, vals = idxs[order], vals[order]

    # Coalesce duplicates
    if idxs.size > 1 and (idxs[1:] == idxs[:-1]).any():
        uniq, start = np.unique(idxs, return_index=True)
        vals = np.add.reduceat(vals, start)
        idxs = uniq

    return idxs, vals


def csr_row_from_tuple(item: Tuple[np.ndarray, np.ndarray], dim: int) -> sp.csr_matrix:
    """Build a single-row CSR matrix from (indices, values) tuple."""
    idxs, vals = _coalesce_sparse_row(item[0], item[1], dim)
    indptr = np.array([0, idxs.size], dtype=np.int64)
    return sp.csr_matrix((vals, idxs, indptr), shape=(1, dim), dtype=np.float32)


def l1_norms_csr(X: sp.csr_matrix) -> np.ndarray:
    """Compute L1 norms for each row of a CSR matrix."""
    return np.asarray(np.abs(X).sum(axis=1)).ravel().astype(np.float64)


def _umap_from_precomputed_knn(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    *,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    spread: float = 1.0,
    random_state: int = None,
    n_epochs: Optional[int] = None,
    negative_sample_rate: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build UMAP embedding from a precomputed kNN graph.

    Uses UMAP's native precomputed_knn parameter for clean integration.
    """
    N = knn_indices.shape[0]

    # UMAP accepts precomputed kNN via tuple: (indices, distances, forest)
    # forest=None signals we don't have an RP forest for additional queries
    precomputed_knn = (
        knn_indices.astype(np.int32),
        knn_dists.astype(np.float32),
        None,  # no random projection forest
    )

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        n_epochs=n_epochs,
        negative_sample_rate=negative_sample_rate,
        verbose=verbose,
        precomputed_knn=precomputed_knn,
        metric="precomputed",  # signals we're using precomputed distances
    )

    # Fit with dummy data; UMAP will use precomputed_knn
    X_dummy = np.zeros((N, 1), dtype=np.float32)
    embedding = reducer.fit_transform(X_dummy)

    return np.asarray(embedding, dtype=np.float32)


# ======================
# Base class
# ======================

@dataclass
class _BaseANN:
    """Base class for ANN indices."""
    dim: int
    space: str
    _meta: dict = field(default_factory=dict)

    def save_index(self, path_prefix: str) -> None:
        raise NotImplementedError

    def load_index(self, path_prefix: str) -> None:
        raise NotImplementedError

    def build_index(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError


# ======================
# ANN-1: MS2Deepscore embeddings (dense cosine, nmslib)
# ======================

class EmbeddingIndex(_BaseANN):
    """
    Dense cosine ANN using nmslib HNSW.

    Methods
    -------
    build_index(vectors, spec_ids, ...)
    build_index_from_sqlite(db, embeddings_table='embeddings', ...)
    query(vector, k)
    save_index / load_index
    """

    def __init__(self, dim: int = 500):
        super().__init__(dim=dim, space="cosinesimil")
        self._index: Optional[nmslib.dist.FloatIndex] = None
        self._ids: Optional[np.ndarray] = None

    def build_index(
        self,
        vectors: np.ndarray,
        spec_ids: Iterable[str],
        *,
        M: int = 16,
        ef_construction: int = 200,
        post_init_ef: int = 200,
    ) -> None:
        """
        Build index from dense vectors.

        Parameters
        ----------
        vectors : np.ndarray
            2D array of shape (N, dim) with float32 vectors.
        spec_ids : Iterable[str]
            Spec IDs of length N.
        M, ef_construction, post_init_ef : int
            HNSW parameters.
        """
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {X.shape}")

        ids = np.asarray(list(spec_ids), dtype=object)
        if len(ids) != len(X):
            raise ValueError("spec_ids length must match number of vectors.")

        self._index = self._create_hnsw_index(
            X, M=M, ef_construction=ef_construction, post_init_ef=post_init_ef, sparse=False
        )
        self._ids = ids
        self._meta = {
            "type": "EmbeddingIndex",
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
        }

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
        Stream embeddings from SQLite and build HNSW index.

        Returns the number of vectors indexed.

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
        """
        cur = db.cursor()

        # Detect dimension
        d = self._detect_dimension(cur, embeddings_table)
        self.dim = d

        # Count and load
        where_clause = _build_where_clause(where_sql)
        total = _count_rows(cur, embeddings_table, where_clause)
        if total == 0:
            raise ValueError(f"No embeddings in {embeddings_table}.")

        # Pre-allocate and fill
        X = np.empty((total, d), dtype=np.float32)
        ids = np.empty(total, dtype=object)

        sql = f"SELECT spec_id, vec FROM {embeddings_table} {where_clause} ORDER BY spec_id ASC"
        cur.execute(sql)

        filled = 0
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            for sid, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32, count=d)
                if vec.size != d:
                    raise ValueError(f"Embedding '{sid}' has {vec.size} dims, expected {d}")
                X[filled] = vec
                ids[filled] = str(sid)
                filled += 1

        cur.close()

        # Handle case where table changed mid-scan
        if filled != total:
            X = X[:filled]
            ids = ids[:filled]
            if filled == 0:
                raise ValueError(f"No embeddings loaded from {embeddings_table}.")

        self._index = self._create_hnsw_index(
            X, M=M, ef_construction=ef_construction, post_init_ef=post_init_ef, sparse=False
        )
        self._ids = ids
        self._meta = {
            "type": "EmbeddingIndex",
            "built_from_sqlite": True,
            "embeddings_table": embeddings_table,
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
        }
        return total

    def _detect_dimension(self, cursor: sqlite3.Cursor, table: str) -> int:
        """Detect embedding dimension from table."""
        dims = [int(r[0]) for r in cursor.execute(f"SELECT DISTINCT d FROM {table}")]
        if not dims:
            raise ValueError(f"No rows in '{table}'.")
        if len(dims) > 1:
            raise ValueError(f"Mixed dimensions in '{table}': {dims}")
        return dims[0]

    def _create_hnsw_index(
        self, data, *, M: int, ef_construction: int, post_init_ef: int, sparse: bool
    ):
        """Create and configure an nmslib HNSW index."""
        dtype = nmslib.DataType.SPARSE_VECTOR if sparse else nmslib.DataType.DENSE_VECTOR
        index = nmslib.init(method="hnsw", space=self.space, data_type=dtype)
        index.addDataPointBatch(data)
        index.createIndex({"M": M, "efConstruction": ef_construction}, print_progress=False)
        index.setQueryTimeParams({"ef": post_init_ef})
        return index

    def query(
        self,
        vectors: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        num_threads: int = 0,
    ) -> List[Tuple[str, float]] | List[List[Tuple[str, float]]]:
        """
        Query for k nearest neighbors.

        Parameters
        ----------
        vectors : np.ndarray
            Either a single vector of shape (dim,) or a batch of shape (N, dim).
        k : int
            Number of neighbors.
        ef : Optional[int]
            Optional per-query ef parameter for HNSW.
        num_threads : int
            Number of threads to use inside nmslib (0 = library default).

        Returns
        -------
        Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]
            - If a single vector is given, returns a list of (spec_id, similarity).
            - If a batch is given, returns a list (per query) of such lists.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")

        X = np.asarray(vectors, dtype=np.float32)

        single = False
        if X.ndim == 1:
            # Single query vector: (dim,) -> (1, dim)
            if X.size != self.dim:
                raise ValueError(f"Query must have dim={self.dim}")
            X = X.reshape(1, -1)
            single = True
        elif X.ndim == 2:
            if X.shape[1] != self.dim:
                raise ValueError(f"Expected shape (N, {self.dim}), got {X.shape}")
        else:
            raise ValueError("vectors must be 1D or 2D array.")

        if ef is not None:
            self._index.setQueryTimeParams({"ef": ef})

        batch_results = self._index.knnQueryBatch(X, k=k, num_threads=num_threads)

        all_out: List[List[Tuple[str, float]]] = []
        for idxs, dists in batch_results:
            idxs = np.asarray(idxs, dtype=np.int64)
            dists = np.asarray(dists, dtype=np.float32)
            sims = 1.0 - dists  # cosine distance -> similarity
            out = [(str(self._ids[i]), float(s)) for i, s in zip(idxs, sims)]
            all_out.append(out)

        return all_out[0] if single else all_out

    def save_index(self, path_prefix: str) -> None:
        if self._index is None:
            raise RuntimeError("Index not built.")

        self._index.saveIndex(path_prefix, save_data=True)
        np.save(f"{path_prefix}.ids.npy", self._ids)

        meta = {**self._meta, "dim": self.dim, "space": self.space}
        with open(f"{path_prefix}.meta.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path_prefix: str) -> None:
        with open(f"{path_prefix}.meta.json") as f:
            meta = json.load(f)

        self.dim = int(meta["dim"])
        self.space = str(meta["space"])
        self._meta = meta
        self._ids = np.load(f"{path_prefix}.ids.npy", allow_pickle=True)

        self._index = nmslib.init(
            method="hnsw", space=self.space, data_type=nmslib.DataType.DENSE_VECTOR
        )
        self._index.loadIndex(path_prefix, load_data=True)


# ======================
# ANN-2: Sparse fingerprints with Tanimoto re-ranking
# ======================

class FingerprintSparseIndex(_BaseANN):
    """
    Sparse fingerprint index with ANN + exact Tanimoto re-ranking.

    Uses nmslib HNSW over sparse cosine for fast candidate retrieval,
    then re-ranks with exact generalized Tanimoto (L1/Jaccard).
    """

    def __init__(self, dim: int = 4096):
        super().__init__(dim=dim, space="cosinesimil_sparse")
        self._index: Optional[nmslib.dist.FloatIndex] = None
        self._comp_ids: Optional[np.ndarray] = None
        self._csr: Optional[sp.csr_matrix] = None
        self._l1: Optional[np.ndarray] = None

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
        """
        Build index from sparse fingerprints.

        Parameters
        ----------
        data : CSR matrix or sequence of (indices, values) tuples
        comp_ids : Compound IDs
        dim : Required if data is sequence of tuples
        keep_csr_for_rerank : Store CSR for exact Tanimoto re-ranking
        compute_l1_for_rerank : Precompute L1 norms for re-ranking
        """
        if isinstance(data, sp.csr_matrix):
            csr = data.astype(np.float32, copy=False)
            D = csr.shape[1]
        else:
            D = dim if dim is not None else self.dim
            csr = tuples_to_csr(data, dim=D)

        if (csr.data < 0).any():
            raise ValueError("Fingerprints must be non-negative for Tanimoto.")

        self.dim = D
        comp_ids = np.asarray(list(comp_ids))
        if len(comp_ids) != csr.shape[0]:
            raise ValueError("comp_ids length must match data rows.")

        # Build ANN index
        index = nmslib.init(
            method="hnsw", space=self.space, data_type=nmslib.DataType.SPARSE_VECTOR
        )
        index.addDataPointBatch(csr)
        index.createIndex({"M": M, "efConstruction": ef_construction}, print_progress=False)
        index.setQueryTimeParams({"ef": post_init_ef})

        self._index = index
        self._comp_ids = comp_ids
        self._csr = csr if keep_csr_for_rerank else None
        self._l1 = l1_norms_csr(csr) if compute_l1_for_rerank else None
        self._meta = {
            "type": "FingerprintSparseIndex",
            "M": M,
            "ef_construction": ef_construction,
            "post_init_ef": post_init_ef,
        }

    def query(
        self,
        query_fp: (
            Tuple[np.ndarray, np.ndarray]
            | sp.csr_matrix
            | Sequence[Tuple[np.ndarray, np.ndarray]]
        ),
        k: int = 10,
        *,
        ef: Optional[int] = None,
        re_rank: bool = True,
        candidate_multiplier: int = 5,
        num_threads: int = 0,
    ) -> List[Tuple[int, float]] | List[List[Tuple[int, float]]]:
        """
        Query for k nearest neighbors.

        Parameters
        ----------
        query_fp :
            - Single query:
                * (indices, values) tuple
                * single-row CSR of shape (1, dim)
            - Batched queries:
                * CSR of shape (N, dim)
                * Sequence of (indices, values) tuples
        k : int
            Number of results per query.
        re_rank : bool
            Use exact Tanimoto re-ranking.
        candidate_multiplier : int
            Fetch k * multiplier candidates for re-ranking.
        num_threads : int
            Number of threads to use inside nmslib (0 = library default).

        Returns
        -------
        Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]
            - For a single query, returns a list of (comp_id, similarity).
            - For multiple queries, returns a list (per query) of such lists.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")

        # -------------------------
        # Normalize input to CSR
        # -------------------------
        single = False

        if isinstance(query_fp, sp.csr_matrix):
            Q = query_fp.astype(np.float32, copy=False)
            if Q.shape[1] != self.dim:
                raise ValueError(f"CSR query must have shape (N, {self.dim})")
            single = Q.shape[0] == 1

        elif isinstance(query_fp, tuple):
            # Single (indices, values)
            Q = csr_row_from_tuple(query_fp, dim=self.dim)
            single = True

        else:
            # Assume sequence of (indices, values) tuples -> batched queries
            Q = tuples_to_csr(query_fp, dim=self.dim)
            single = Q.shape[0] == 1

        if (Q.data < 0).any():
            raise ValueError("Query must be non-negative for Tanimoto.")

        # Handle completely empty queries quickly
        row_nnz = Q.indptr[1:] - Q.indptr[:-1]
        if row_nnz.sum() == 0:
            if single:
                return []
            return [[] for _ in range(Q.shape[0])]

        if ef is not None:
            self._index.setQueryTimeParams({"ef": ef})

        fetch = max(k, k * candidate_multiplier)

        # -------------------------
        # ANN search for all queries
        # -------------------------
        batch_results = self._index.knnQueryBatch(Q, k=fetch, num_threads=num_threads)

        # -------------------------
        # No re-ranking: cosine sims only
        # -------------------------
        if not re_rank or self._csr is None or self._l1 is None:
            all_out: List[List[Tuple[int, float]]] = []

            for qi, (idxs, dists) in enumerate(batch_results):
                if row_nnz[qi] == 0:
                    all_out.append([])
                    continue

                idxs = np.asarray(idxs, dtype=np.int64)
                dists = np.asarray(dists, dtype=np.float32)

                sims = 1.0 - dists
                out = [
                    (self._comp_ids[i], float(s))
                    for i, s in zip(idxs[:k], sims[:k])
                ]
                all_out.append(out)

            return all_out[0] if single else all_out

        # -------------------------
        # Exact Tanimoto re-ranking
        # -------------------------
        all_out: List[List[Tuple[int, float]]] = []

        for qi, (idxs, dists) in enumerate(batch_results):
            if row_nnz[qi] == 0:
                all_out.append([])
                continue

            idxs = np.asarray(idxs, dtype=np.int64)

            q_row = Q[qi]
            Y = self._csr[idxs]
            tan = tanimoto_l1_query_vs_block(
                q_row,
                Y,
                sum1=float(q_row.sum()),
                sumsY=self._l1[idxs],
            )

            order = np.argsort(-tan)[:k]
            out = [
                (self._comp_ids[idxs[i]], float(tan[i]))
                for i in order
            ]
            all_out.append(out)

        return all_out[0] if single else all_out

    def _normalize_query(self, query_fp) -> sp.csr_matrix:
        """Convert query to single-row CSR and validate."""
        if isinstance(query_fp, sp.csr_matrix):
            if query_fp.shape[0] != 1 or query_fp.shape[1] != self.dim:
                raise ValueError(f"CSR query must have shape (1, {self.dim})")
            q = query_fp
        else:
            q = csr_row_from_tuple(query_fp, dim=self.dim)

        if (q.data < 0).any():
            raise ValueError("Query must be non-negative for Tanimoto.")
        return q

    def compute_dr_coordinates(
        self,
        *,
        n_neighbors: int = 25,
        n_components: int = 2,
        min_dist: float = 0.2,
        spread: float = 1.0,
        random_state: int = None,
        n_epochs: Optional[int] = None,
        negative_sample_rate: int = 5,
        ef: Optional[int] = None,
        candidate_multiplier: int = 5,
        use_exact_tanimoto: bool = True,
        verbose: bool = False,
        batch_size: int = 2048,
        num_threads: int = 0,
    ) -> np.ndarray:
        """
        Compute UMAP embedding coordinates from the index.

        Builds kNN graph using ANN (optionally with Tanimoto re-ranking),
        then runs UMAP on the precomputed graph.
        """
        if self._index is None or self._comp_ids is None:
            raise RuntimeError("Index not built or loaded.")
        if self._csr is None:
            raise RuntimeError("CSR not available; use keep_csr_for_rerank=True")
        if use_exact_tanimoto and self._l1 is None:
            raise RuntimeError("L1 norms not available; use compute_l1_for_rerank=True")

        if ef is not None:
            self._index.setQueryTimeParams({"ef": ef})

        N = len(self._comp_ids)
        fetch = max(n_neighbors, n_neighbors * candidate_multiplier)

        knn_idx = np.empty((N, n_neighbors), dtype=np.int32)
        knn_dist = np.empty((N, n_neighbors), dtype=np.float32)

        for start in tqdm(range(0, N, batch_size), disable=not verbose, desc="Building kNN"):
            end = min(start + batch_size, N)
            results = self._index.knnQueryBatch(
                self._csr[start:end], k=fetch, num_threads=num_threads
            )

            for off, (idxs, dists) in enumerate(results):
                i = start + off
                idxs = np.asarray(idxs, dtype=np.int32)
                dists = np.asarray(dists, dtype=np.float32)

                if use_exact_tanimoto:
                    # Re-rank with exact Tanimoto
                    q = self._csr[i]
                    sims = tanimoto_l1_query_vs_block(
                        q, self._csr[idxs], sum1=float(q.sum()), sumsY=self._l1[idxs]
                    )
                    order = np.argsort(-sims)
                    idxs, dists = idxs[order], (1.0 - sims[order]).astype(np.float32)

                # Ensure self is in neighbors
                idxs, dists = self._ensure_self_neighbor(i, idxs, dists, n_neighbors)
                knn_idx[i] = idxs
                knn_dist[i] = dists

        return _umap_from_precomputed_knn(
            knn_idx, knn_dist,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            spread=spread,
            random_state=random_state,
            n_epochs=n_epochs,
            negative_sample_rate=negative_sample_rate,
            verbose=verbose,
        )

    @staticmethod
    def _ensure_self_neighbor(
        i: int, idxs: np.ndarray, dists: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure point i is in its own neighbor list."""
        if i not in idxs[:k]:
            # Insert self at position with max distance
            j = int(np.argmax(dists[:k]))
            idxs[j], dists[j] = i, 0.0

        order = np.argsort(dists)[:k]
        return idxs[order], dists[order]

    def save_index(self, path_prefix: str) -> None:
        if self._index is None:
            raise RuntimeError("Index not built.")

        # Also save data so that load_index(..., load_data=True) works
        self._index.saveIndex(f"{path_prefix}.nmslib", save_data=True)
        np.save(f"{path_prefix}.ids.npy", self._comp_ids)

        meta = {**self._meta, "dim": int(self.dim), "space": str(self.space)}
        with open(f"{path_prefix}.meta.json", "w") as f:
            json.dump(meta, f)

        if self._csr is not None:
            sp.save_npz(f"{path_prefix}.csr.npz", self._csr, compressed=True)
        if self._l1 is not None:
            np.save(f"{path_prefix}.l1.npy", self._l1)

    def load_index(self, path_prefix: str) -> None:
        with open(f"{path_prefix}.meta.json") as f:
            meta = json.load(f)

        self.dim = int(meta["dim"])
        self.space = str(meta["space"])
        self._meta = meta

        self._comp_ids = np.load(f"{path_prefix}.ids.npy", allow_pickle=False)

        self._index = nmslib.init(
            method="hnsw", space=self.space, data_type=nmslib.DataType.SPARSE_VECTOR
        )
        self._index.loadIndex(f"{path_prefix}.nmslib", load_data=True)

        csr_path = f"{path_prefix}.csr.npz"
        l1_path = f"{path_prefix}.l1.npy"
        self._csr = sp.load_npz(csr_path).astype(np.float32) if os.path.exists(csr_path) else None
        self._l1 = np.load(l1_path, allow_pickle=False) if os.path.exists(l1_path) else None


# ======================
# Helper functions
# ======================

def _build_where_clause(where_sql: Optional[str]) -> str:
    """Normalize WHERE clause input."""
    if not where_sql:
        return ""
    where_sql = where_sql.strip()
    return where_sql if where_sql.upper().startswith("WHERE") else f"WHERE {where_sql}"


def _count_rows(cursor: sqlite3.Cursor, table: str, where_clause: str) -> int:
    """Count rows in table with optional WHERE clause."""
    cursor.execute(f"SELECT COUNT(1) FROM {table} {where_clause}")
    return int(cursor.fetchone()[0])
