import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from ms2deepscore.models import load_model as _ms2ds_load_model
from ms2query.data_processing import compute_spectra_embeddings


# ------------ helpers ------------

_NUMERIC_FIELDS = {"precursor_mz"}  # stored as REAL
_TEXT_FIELDS = {
    "ionmode", "smiles", "inchikey", "inchi", "name",
    "instrument_type", "adduct", "collision_energy",
}

def _as_float32_bytes(a: np.ndarray) -> bytes:
    if a is None:
        return b""
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a.tobytes(order="C")

def _from_float32_bytes(b: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.float32, count=n)
    return np.array(arr, copy=True)

def _from_float32_bytes_known_dim(b: bytes, d: int) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.float32, count=d)
    if arr.size != d:
        raise ValueError(f"Expected {d} floats in embedding blob, found {arr.size}.")
    return np.array(arr, copy=True)

def _normalize_metadata(md: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in fields:
        val = md.get(key, None)
        if key in _NUMERIC_FIELDS:
            if val is None or (isinstance(val, float) and (np.isnan(val))):
                out[key] = None
            else:
                try:
                    out[key] = float(val)
                except Exception:
                    out[key] = None
        else:
            out[key] = None if val in (None, "") else str(val)
    return out


# ------------ main class ------------

@dataclass
class SpectralDatabase:
    sqlite_path: str
    table: str = "spectra"
    metadata_fields: List[str] = field(
        default_factory=lambda: [
            "precursor_mz",
            "ionmode",
            "smiles",
            "inchikey",
            "inchi",
            "name",
            "instrument_type",
            "adduct",
            "collision_energy",
        ]
    )
    spectrum_sum_normalization_for_embedding: bool = True
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _ms2ds_model_path: Optional[str] = field(default=None, repr=False)
    _ms2ds_model: Any = field(default=None, repr=False)

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ---------- public API ----------

    def add_spectra(self, spectra: List[Spectrum]) -> List[str]:
        """Add spectra to the database. Returns spec_ids (spectrum hashes)."""
        if not spectra:
            return []

        cur = self._conn.cursor()
        # Bulk-load speed PRAGMAs (safe for single-user/batch ingest)
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
        """)
        cur.execute("BEGIN")

        spec_ids: List[str] = []

        # Build INSERT with explicit spec_id; use OR IGNORE to avoid duplicate rows
        col_list = ["spec_id", "mz_blob", "intensity_blob", "n_peaks"] + self.metadata_fields
        placeholders = ",".join("?" for _ in col_list)
        sql = f"INSERT OR IGNORE INTO {self.table} ({', '.join(col_list)}) VALUES ({placeholders})"

        try:
            for sp in spectra:
                mz = sp.mz
                intens = sp.intensities
                if mz.shape[0] != intens.shape[0]:
                    raise ValueError("m/z and intensity arrays have different lengths.")

                n = int(mz.shape[0])
                md_norm = _normalize_metadata(sp.metadata, self.metadata_fields)

                spec_hash = str(sp.spectrum_hash())  # <- string spec_id
                values = [
                    spec_hash,
                    _as_float32_bytes(mz),
                    _as_float32_bytes(intens),
                    n,
                ] + [md_norm[k] for k in self.metadata_fields]

                cur.execute(sql, values)
                # Whether inserted or ignored as duplicate, we return the hash
                spec_ids.append(spec_hash)

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return spec_ids

    def ids(self) -> List[str]:
        """Return all spec_ids (hash strings) in the database."""
        cur = self._conn.cursor()
        rows = cur.execute(f"SELECT spec_id FROM {self.table}").fetchall()
        return [str(row["spec_id"]) for row in rows]

    def get_spectra_by_ids(self, spec_ids: List[str]) -> List[Spectrum]:
        """Retrieve full Spectrum objects for given spec_ids (order preserved, missing IDs skipped)."""
        rows = self._fetch_rows_by_ids(
            spec_ids, cols="spec_id, mz_blob, intensity_blob, n_peaks, " + ", ".join(self.metadata_fields))
        by_id = {row["spec_id"]: row for row in rows}

        result: List[Spectrum] = []
        for sid in spec_ids:
            row = by_id.get(sid)
            if row is None:
                continue
            n = int(row["n_peaks"])
            mz = _from_float32_bytes(row["mz_blob"], n)
            inten = _from_float32_bytes(row["intensity_blob"], n)
            md = {k: row[k] for k in self.metadata_fields}
            md["spec_id"] = sid
            result.append(Spectrum(mz=mz, intensities=inten, metadata=md))
        return result

    def get_fragments_by_ids(self, spec_ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve (mz, intensity) arrays for given spec_ids.

        Order is preserved with respect to `spec_ids`.
        Missing IDs are skipped.
        """
        cols = "spec_id, mz_blob, intensity_blob, n_peaks"
        rows = self._fetch_rows_by_ids(spec_ids, cols=cols)
        by_id = {row["spec_id"]: row for row in rows}

        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for sid in spec_ids:
            row = by_id.get(sid)
            if row is None:
                continue
            n = int(row["n_peaks"])
            mz = _from_float32_bytes(row["mz_blob"], n)
            inten = _from_float32_bytes(row["intensity_blob"], n)
            out.append((mz, inten))
        return out

    def get_metadata_by_ids(self, spec_ids: List[str]) -> pd.DataFrame:
        """
        Retrieve metadata for given spec_ids.

        Returns a DataFrame with **one row per requested spec_id** in the same
        order as `spec_ids`. If a spec_id is not present in the database, a row
        with that spec_id and metadata columns set to None/NaN is returned.
        """
        cols = ["spec_id"] + self.metadata_fields
        if not spec_ids:
            return pd.DataFrame(columns=cols)

        rows = self._fetch_rows_by_ids(spec_ids, cols=", ".join(cols))
        by_id = {row["spec_id"]: row for row in rows}

        records: List[Dict[str, Any]] = []
        for sid in spec_ids:
            row = by_id.get(sid)
            if row is None:
                rec = {"spec_id": sid}
                rec.update({k: None for k in self.metadata_fields})
            else:
                rec = {"spec_id": sid}
                for k in self.metadata_fields:
                    rec[k] = row[k]
            records.append(rec)

        return pd.DataFrame.from_records(records, columns=cols)

    def sql_query(self, query: str) -> pd.DataFrame:
        """Run a raw SQL SELECT and return a DataFrame."""
        return pd.read_sql_query(query, self._conn)

    def ensure_embeddings_schema(self, table: str = "embeddings") -> None:
        """
        Ensure an embeddings table exists with:
          - spec_id TEXT PRIMARY KEY
          - d      INTEGER (dimension)
          - vec    BLOB (float32[d], raw)
        """
        cur = self._conn.cursor()
        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS {table}(
                spec_id TEXT PRIMARY KEY NOT NULL,
                d       INTEGER NOT NULL,
                vec     BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_{table}_d ON {table}(d);
        """)
        self._conn.commit()

    def load_ms2deepscore_model(self, model_path: str):
        """
        Lazy-load and cache the MS2DeepScore model, keeping path for reproducible calls.
        """
        if self._ms2ds_model is None or (self._ms2ds_model_path != model_path):
            model = _ms2ds_load_model(model_path)
            model.eval()
            self._ms2ds_model = model
            self._ms2ds_model_path = model_path
        return self._ms2ds_model

    def compute_embeddings_to_sqlite(
        self,
        model_path: str,
        *,
        spectra_table: Optional[str] = None,
        embeddings_table: str = "embeddings",
        batch_rows: int = 1024,
        only_missing: bool = True,
        commit_every: int = 0,
    ) -> int:
        """
        Compute MS2DeepScore embeddings for rows in `spectra_table` and write to `embeddings_table`.

        - Uses `matchms.Spectrum` objects reconstructed from the stored peaks & metadata.
        - Stores raw float32 vectors (no extra header) with their dimension `d`.
        """
        spectra_table = spectra_table or self.table
        self._ensure_schema()  # spectra schema
        self.ensure_embeddings_schema(embeddings_table)

        cur = self._conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        if only_missing:
            query = f"""
                SELECT s.spec_id, s.mz_blob, s.intensity_blob, s.n_peaks,
                       s.precursor_mz, s.ionmode, s.charge
                FROM {spectra_table} s
                LEFT JOIN {embeddings_table} e ON s.spec_id = e.spec_id
                WHERE e.spec_id IS NULL
                ORDER BY s.spec_id ASC;
            """
        else:
            query = f"""
                SELECT spec_id, mz_blob, intensity_blob, n_peaks,
                       precursor_mz, ionmode, charge
                FROM {spectra_table}
                ORDER BY spec_id ASC;
            """
        cur.execute(query)

        model = self.load_ms2deepscore_model(model_path)

        inserted = 0
        buf: List[
            Tuple[str, bytes, bytes, int, float, str, Optional[int]]
        ] = []
        done_since_commit = 0

        def flush(batch) -> int:
            if not batch:
                return 0
            specs: List[Spectrum] = []
            sids: List[str] = []
            for sid, mz_blob, it_blob, n_peaks, prec_mz, ionmode, charge in batch:
                mz = _from_float32_bytes(mz_blob, int(n_peaks))
                it = _from_float32_bytes(it_blob, int(n_peaks))
                spectrum = Spectrum(
                    mz=mz,
                    intensities=it,
                    metadata={
                        "precursor_mz": float(prec_mz) if prec_mz is not None else None,
                        "ionmode": ionmode,
                        "charge": charge,
                        "spec_id": sid,
                    },
                )
                specs.append(spectrum)
                sids.append(sid)

            embeddings = compute_spectra_embeddings(
                model,
                specs,
                normalize_spectrum=self.spectrum_sum_normalization_for_embedding,
            )
            dim = int(embeddings.shape[1])
            q = (
                f"INSERT OR REPLACE INTO {embeddings_table} "
                f"(spec_id, d, vec) VALUES (?, ?, ?);"
            )
            with self._conn:
                for sid, embedding in zip(sids, embeddings):
                    self._conn.execute(
                        q,
                        (sid, dim, sqlite3.Binary(_as_float32_bytes(embedding))),
                    )
            return len(batch)

        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            buf.extend(rows)
            while len(buf) >= batch_rows:
                inserted += flush(buf[:batch_rows])
                buf = buf[batch_rows:]
                done_since_commit += batch_rows
                if commit_every and done_since_commit >= commit_every:
                    self._conn.commit()
                    done_since_commit = 0

        inserted += flush(buf)
        return inserted

    def get_embeddings(
        self,
        spec_ids: Optional[List[str]] = None,
        *,
        embeddings_table: str = "embeddings",
        normalized: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch embeddings by spec_id (or all if ids=None).
        Returns (ids[str], embeddings[float32 of shape (n, d)]).
        If normalized=True, L2-normalize (recommended for cosine).
        """
        cur = self._conn.cursor()
        if spec_ids is None:
            cur.execute(f"SELECT spec_id, d, vec FROM {embeddings_table} ORDER BY spec_id ASC;")
        else:
            placeholders = ",".join("?" for _ in spec_ids)
            cur.execute(
                f"""SELECT spec_id, d, vec FROM {embeddings_table}
                WHERE spec_id IN ({placeholders}) ORDER BY spec_id ASC;""",
                spec_ids)

        sids: List[str] = []
        vecs: List[np.ndarray] = []
        d_first: Optional[int] = None
        for sid, d, blob in cur:
            d = int(d)
            if d_first is None:
                d_first = d
            elif d_first != d:
                raise ValueError(f"Mixed embedding dimensions in {embeddings_table}: {d_first} vs {d}")
            sids.append(str(sid))
            vecs.append(_from_float32_bytes_known_dim(blob, d))
        if not vecs:
            return np.empty((0,), dtype=str), np.empty((0, 0), dtype=np.float32)

        X = np.vstack(vecs).astype(np.float32, copy=False)
        if normalized:
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n = np.maximum(n, 1e-12)
            X = X / n
        return np.asarray(sids, dtype=object), X

    def get_embedding_for_id(
        self,
        spec_id: str,
        *,
        embeddings_table: str = "embeddings",
        normalized: bool = True,
    ) -> Optional[np.ndarray]:
        ids, X = self.get_embeddings([spec_id], embeddings_table=embeddings_table, normalized=normalized)
        return X[0] if X.shape[0] else None

    # expose raw connection for ANN builders
    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn
    # ---------- internal ----------

    def _fetch_rows_by_ids(self, spec_ids: List[str], cols: str) -> List[sqlite3.Row]:
        if not spec_ids:
            return []
        placeholders = ",".join("?" for _ in spec_ids)
        sql = f"SELECT {cols} FROM {self.table} WHERE spec_id IN ({placeholders})"
        cur = self._conn.cursor()
        return cur.execute(sql, spec_ids).fetchall()

    def _ensure_schema(self):
        cur = self._conn.cursor()
        # Build metadata columns
        md_cols_sql = []
        for k in self.metadata_fields:
            if k in _NUMERIC_FIELDS:
                md_cols_sql.append(f"{k} REAL")
            else:
                md_cols_sql.append(f"{k} TEXT")
        md_cols_clause = ", ".join(md_cols_sql)

        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS {self.table}(
                spec_id        TEXT PRIMARY KEY NOT NULL,
                mz_blob        BLOB NOT NULL,
                intensity_blob BLOB NOT NULL,
                n_peaks        INTEGER NOT NULL,
                {md_cols_clause}
            );
            CREATE INDEX IF NOT EXISTS idx_{self.table}_inchikey ON {self.table}(inchikey);
            CREATE INDEX IF NOT EXISTS idx_{self.table}_precursor_mz ON {self.table}(precursor_mz);
        """)
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
