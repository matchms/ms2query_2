import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum


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
    metadata_fields: List[str] = field(default_factory=lambda: [
        "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
        "instrument_type", "adduct", "collision_energy"
    ])
    _conn: sqlite3.Connection = field(init=False, repr=False)

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
        cur.executescript("""
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

    def get_spectra_by_ids(self, specIDs: List[str]) -> List[Spectrum]:
        """Retrieve full Spectrum objects for given specIDs (order preserved, missing IDs skipped)."""
        rows = self._fetch_rows_by_ids(
            specIDs, cols="spec_id, mz_blob, intensity_blob, n_peaks, " + ", ".join(self.metadata_fields))
        by_id = {row["spec_id"]: row for row in rows}

        result: List[Spectrum] = []
        for sid in specIDs:
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

    def get_fragments_by_ids(self, specIDs: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve (mz, intensity) arrays for given specIDs (order preserved, missing IDs skipped)."""
        rows = self._fetch_rows_by_ids(specIDs, cols="spec_id, mz_blob, intensity_blob, n_peaks")
        by_id = {row["spec_id"]: row for row in rows}

        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for sid in specIDs:
            row = by_id.get(sid)
            if row is None:
                continue
            n = int(row["n_peaks"])
            mz = _from_float32_bytes(row["mz_blob"], n)
            inten = _from_float32_bytes(row["intensity_blob"], n)
            out.append((mz, inten))
        return out

    def get_metadata_by_ids(self, specIDs: List[str]) -> pd.DataFrame:
        """Retrieve metadata for given specIDs (order preserved)."""
        cols = ["spec_id"] + self.metadata_fields
        rows = self._fetch_rows_by_ids(specIDs, cols=", ".join(cols))
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            order = {sid: i for i, sid in enumerate(specIDs)}
            df["__order"] = df["spec_id"].map(order)
            df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        return df

    def sql_query(self, query: str) -> pd.DataFrame:
        """Run a raw SQL SELECT and return a DataFrame."""
        return pd.read_sql_query(query, self._conn)

    # ---------- internal ----------

    def _fetch_rows_by_ids(self, specIDs: List[str], cols: str) -> List[sqlite3.Row]:
        if not specIDs:
            return []
        placeholders = ",".join("?" for _ in specIDs)
        sql = f"SELECT {cols} FROM {self.table} WHERE spec_id IN ({placeholders})"
        cur = self._conn.cursor()
        return cur.execute(sql, specIDs).fetchall()

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
