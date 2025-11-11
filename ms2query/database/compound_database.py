import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ms2query.data_processing import compute_morgan_fingerprints, inchikey14_from_full
from ms2query.database.database_utils import decode_dense_fp, decode_sparse_fp, encode_dense_fp, encode_sparse_fp


# ==================================================
# Compound database (compounds table) in SQLite
# ==================================================

DenseFP = np.ndarray
SparseFP = Tuple[np.ndarray, Optional[np.ndarray]]  # (bits, counts)
AnyFP = Union[DenseFP, SparseFP]

UPSERT_SQL = """
INSERT INTO {table} (
    comp_id, smiles, inchi, inchikey,
    fingerprint_bits, fingerprint_counts, fingerprint_dense,
    fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype,
    classyfire_class, classyfire_superclass
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(comp_id) DO UPDATE SET
    smiles = COALESCE(excluded.smiles, {table}.smiles),
    inchi = COALESCE(excluded.inchi, {table}.inchi),
    inchikey = COALESCE(excluded.inchikey, {table}.inchikey),
    fingerprint_bits = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0) > 0
                                THEN excluded.fingerprint_bits ELSE {table}.fingerprint_bits END,
    fingerprint_counts = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                                THEN excluded.fingerprint_counts ELSE {table}.fingerprint_counts END,
    fingerprint_dense = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_dense),0) > 0
                                THEN excluded.fingerprint_dense ELSE {table}.fingerprint_dense END,
    fp_nbits = COALESCE(excluded.fp_nbits, {table}.fp_nbits),
    fp_radius = COALESCE(excluded.fp_radius, {table}.fp_radius),
    fp_sparse = COALESCE(excluded.fp_sparse, {table}.fp_sparse),
    fp_count = COALESCE(excluded.fp_count, {table}.fp_count),
    fp_dtype = COALESCE(excluded.fp_dtype, {table}.fp_dtype),
    classyfire_class = COALESCE(excluded.classyfire_class, {table}.classyfire_class),
    classyfire_superclass = COALESCE(excluded.classyfire_superclass, {table}.classyfire_superclass)
"""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS {table}(
    comp_id               TEXT PRIMARY KEY,          -- inchikey14
    smiles                TEXT,
    inchi                 TEXT,
    inchikey              TEXT UNIQUE,

    -- sparse storage (pair)
    fingerprint_bits      BLOB,
    fingerprint_counts    BLOB,

    -- dense storage
    fingerprint_dense     BLOB,

    -- FP metadata
    fp_nbits              INTEGER,
    fp_radius             INTEGER,
    fp_sparse             INTEGER,
    fp_count              INTEGER,
    fp_dtype              TEXT,

    classyfire_class      TEXT,
    classyfire_superclass TEXT
);
CREATE INDEX IF NOT EXISTS idx_{table}_smiles ON {table}(smiles);
CREATE INDEX IF NOT EXISTS idx_{table}_inchi  ON {table}(inchi);
"""


@dataclass
class CompoundDatabase:
    """
    SQLite-based compound database with sparse fingerprint storage.
    Stores compounds identified by inchikey14, with optional metadata and molecular fingerprints.

    Can store Morgan fingerprints in any of 4 modes:
      - sparse/binary   : bits only
      - sparse/count    : bits + counts
      - dense/binary    : float32 vector of 0/1
      - dense/count     : float32 vector of counts

    Attributes
    ----------
    sqlite_path : str
        Path to the SQLite database file.
    table : str
        Datbase table name (e.g., "compounds", "reference_compounds")
    compound_fields : List[str]
        List of metadata fields to store for each compound.
    fingerprint_radius : int
        Radius for Morgan fingerprint computation (used in backfill).
    fingerprint_sparse : bool
        Whether to store fingerprints as sparse (True) or dense (False) (used in backfill).
    fingerprint_count : bool
        Whether to store count-based (True) or binary (False) fingerprints (used in backfill).
    """
    sqlite_path: str
    table: str = "compounds"
    compound_fields: List[str] = field(default_factory=lambda: [
        "smiles", "inchi", "inchikey", "classyfire_class", "classyfire_superclass"
    ])
    # Defaults for backfill/metadata
    fingerprint_radius: int = 9
    fingerprint_sparse: bool = True
    fingerprint_count: bool = True
    fingerprint_nbits: int = 4096
    fingerprint_dtype_dense: str = "float32"  # for dense storage
    _conn: sqlite3.Connection = field(init=False, repr=False)

    # ---------------- lifecycle ----------------

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    @contextmanager
    def _tx(self):
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            yield cur
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL.format(table=self.table))

        # Add missing columns for legacy DBs (idempotent)
        existing = {r[1] for r in cur.execute(f"PRAGMA table_info({self.table})").fetchall()}
        desired = {
            "smiles": "TEXT",
            "inchi": "TEXT",
            "inchikey": "TEXT",
            "fingerprint_bits": "BLOB",
            "fingerprint_counts": "BLOB",
            "fingerprint_dense": "BLOB",
            "fp_nbits": "INTEGER",
            "fp_radius": "INTEGER",
            "fp_sparse": "INTEGER",
            "fp_count": "INTEGER",
            "fp_dtype": "TEXT",
            "classyfire_class": "TEXT",
            "classyfire_superclass": "TEXT",
        }
        for name, typ in desired.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE {self.table} ADD COLUMN {name} {typ}")
        self._conn.commit()

    # ---------------- helpers ----------------

    def _pack_fp_for_write(
        self,
        fp: Optional[AnyFP],
        *,
        sparse: Optional[bool],
        count: Optional[bool],
        radius: Optional[int],
        nbits: Optional[int],
        dtype_dense: Optional[str],
    ) -> Dict[str, Any]:
        """Map an optional fingerprint + settings to DB column payload."""
        cols = {
            "fingerprint_bits": b"",
            "fingerprint_counts": b"",
            "fingerprint_dense": b"",
            "fp_nbits": nbits or self.fingerprint_nbits,
            "fp_radius": radius or self.fingerprint_radius,
            "fp_sparse": 1 if (self.fingerprint_sparse if sparse is None else sparse) else 0,
            "fp_count": 1 if (self.fingerprint_count  if count  is None else count)  else 0,
            "fp_dtype": dtype_dense or self.fingerprint_dtype_dense,
        }
        if fp is None:
            return cols

        if cols["fp_sparse"]:
            # sparse: fp can be (bits, counts) or just bits
            if isinstance(fp, tuple):
                bits, counts = fp
                b_blob, c_blob = encode_sparse_fp(bits, counts)
            else:
                b_blob, c_blob = encode_sparse_fp(fp, None)  # type: ignore[arg-type]
            cols.update(fingerprint_bits=b_blob, fingerprint_counts=c_blob, fingerprint_dense=b"")
        else:
            if isinstance(fp, tuple):
                raise ValueError("Dense fingerprint must be a single vector, not (bits, counts).")
            cols.update(fingerprint_dense=encode_dense_fp(fp), fingerprint_bits=b"", fingerprint_counts=b"")  # type: ignore[arg-type]
        return cols

    def _row_to_fp(self, row: sqlite3.Row):
        """Convert row blobs + metadata into a natural Python fingerprint type."""
        dense_blob  = row["fingerprint_dense"] or b""
        bits_blob   = row["fingerprint_bits"] or b""
        counts_blob = row["fingerprint_counts"] or b""

        if dense_blob:
            return decode_dense_fp(dense_blob, dtype=row["fp_dtype"] or "float32")

        if bits_blob or counts_blob:
            bits, counts = decode_sparse_fp(bits_blob, counts_blob)
            return bits if counts.size == 0 else (bits, counts)

        return None

    # ---------------- UPSERTS ----------------

    def _upsert_tuple(
        self,
        comp_id: str,
        *,
        smiles: Optional[str],
        inchi: Optional[str],
        inchikey: Optional[str],
        classyfire_class: Optional[str],
        classyfire_superclass: Optional[str],
        cols: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            comp_id, smiles, inchi, inchikey,
            cols["fingerprint_bits"], cols["fingerprint_counts"], cols["fingerprint_dense"],
            cols["fp_nbits"], cols["fp_radius"], cols["fp_sparse"], cols["fp_count"], cols["fp_dtype"],
            classyfire_class, classyfire_superclass,
        )

    def upsert_compound(
        self,
        *,
        smiles: Optional[str] = None,
        inchi: Optional[str] = None,
        inchikey: Optional[str] = None,
        classyfire_class: Optional[str] = None,
        classyfire_superclass: Optional[str] = None,
        fingerprint: Optional[AnyFP] = None,
        fp_sparse: Optional[bool] = None,
        fp_count: Optional[bool] = None,
        fp_radius: Optional[int] = None,
        fp_nbits: Optional[int] = None,
        fp_dtype: Optional[str] = None,
    ) -> str:
        if not inchikey:
            raise ValueError("inchikey is required to form comp_id (inchikey14).")
        comp_id = inchikey14_from_full(inchikey)
        if not comp_id:
            raise ValueError(f"Invalid InChIKey: {inchikey}")

        cols = self._pack_fp_for_write(
            fingerprint,
            sparse=fp_sparse, count=fp_count,
            radius=fp_radius, nbits=fp_nbits, dtype_dense=fp_dtype,
        )
        with self._tx() as cur:
            cur.execute(UPSERT_SQL.format(table=self.table), self._upsert_tuple(
                comp_id,
                smiles=smiles, inchi=inchi, inchikey=inchikey,
                classyfire_class=classyfire_class,
                classyfire_superclass=classyfire_superclass,
                cols=cols,
            ))
        return comp_id

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> List[str]:
        comp_ids: List[str] = []
        payloads: List[Tuple[Any, ...]] = []

        for r in rows:
            inchikey = r.get("inchikey")
            if not inchikey:
                raise ValueError("Each row must contain 'inchikey'.")
            comp_id = inchikey14_from_full(inchikey)
            if not comp_id:
                raise ValueError(f"Invalid InChIKey: {inchikey}")
            comp_ids.append(comp_id)

            cols = self._pack_fp_for_write(
                r.get("fingerprint"),
                sparse=r.get("fp_sparse"),
                count=r.get("fp_count"),
                radius=r.get("fp_radius"),
                nbits=r.get("fp_nbits"),
                dtype_dense=r.get("fp_dtype"),
            )
            payloads.append(self._upsert_tuple(
                comp_id,
                smiles=r.get("smiles"),
                inchi=r.get("inchi"),
                inchikey=inchikey,
                classyfire_class=r.get("classyfire_class"),
                classyfire_superclass=r.get("classyfire_superclass"),
                cols=cols,
            ))

        if payloads:
            with self._tx() as cur:
                cur.executemany(UPSERT_SQL.format(table=self.table), payloads)
        return comp_ids

    def upsert_metadata_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        colmap: Optional[Dict[str, str]] = None,
        staging_table: str = "_staging_compounds",
    ) -> dict:
        """Load/update metadata (no fingerprints) via a staging table."""
        if df is None or df.empty:
            return {"rows": 0, "valid": 0, "inserted_or_updated": 0, "skipped_no_or_bad_inchikey": 0}

        # Column mapping
        default_map = {
            "inchikey": "inchikey",
            "smiles": "smiles",
            "inchi": "inchi",
            "classyfire_class": "classyfire_class",
            "classyfire_superclass": "classyfire_superclass",
        }
        cmap = {k: (colmap.get(k) if colmap and k in colmap else v) for k, v in default_map.items()}
        if cmap["inchikey"] not in df.columns:
            raise ValueError("DataFrame must contain an 'inchikey' column (or provide colmap).")

        # Build compact frame
        work = pd.DataFrame({"inchikey": df[cmap["inchikey"]].astype(str)})
        work["comp_id"] = work["inchikey"].map(inchikey14_from_full)

        valid_mask = work["comp_id"].notna() & work["comp_id"].astype(str).str.len().eq(14)
        skipped = int((~valid_mask).sum())
        work = work.loc[valid_mask, ["comp_id", "inchikey"]].copy()

        for k in ("smiles", "inchi", "classyfire_class", "classyfire_superclass"):
            src = cmap[k]
            work[k] = df[src] if src in df.columns else None

        work = work.drop_duplicates(subset=["comp_id"], keep="last").reset_index(drop=True)

        # Stage + upsert
        work.to_sql(staging_table, self._conn, if_exists="replace", index=False)
        with self._tx() as cur:
            cur.execute(f"""
                INSERT INTO {self.table} (
                    comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
                )
                SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
                FROM {staging_table}
                ON CONFLICT(comp_id) DO UPDATE SET
                    smiles                = COALESCE(excluded.smiles,                {self.table}.smiles),
                    inchi                 = COALESCE(excluded.inchi,                 {self.table}.inchi),
                    inchikey              = COALESCE(excluded.inchikey,              {self.table}.inchikey),
                    classyfire_class      = COALESCE(excluded.classyfire_class,      {self.table}.classyfire_class),
                    classyfire_superclass = COALESCE(excluded.classyfire_superclass, {self.table}.classyfire_superclass)
            """)
            affected = cur.rowcount or 0
            cur.execute(f"DROP TABLE IF EXISTS {staging_table}")

        return {
            "rows": int(len(df)),
            "valid": int(len(work)),
            "inserted_or_updated": int(affected),
            "skipped_no_or_bad_inchikey": int(skipped),
        }

    # ---------------- READ ----------------

    def get_fingerprint(self, comp_id: str):
        """Return fingerprint for a given comp_id.
        """
        row = self._conn.execute(f"""
            SELECT fingerprint_bits, fingerprint_counts, fingerprint_dense,
                   fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype
            FROM {self.table}
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        return self._row_to_fp(row) if row else None

    def get_fingerprints(self, comp_id_list: List[str]):
        """
        Return list of fingerprints for the given comp_id_list.
        The order of the returned list matches the order of comp_id_list.
        """
        if not comp_id_list:
            return []
        placeholders = ",".join("?" for _ in comp_id_list)
        rows = self._conn.execute(f"""
            SELECT comp_id, fingerprint_bits, fingerprint_counts, fingerprint_dense,
                   fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype
            FROM {self.table}
            WHERE comp_id IN ({placeholders})
        """, comp_id_list).fetchall()
        by_id = {r["comp_id"]: self._row_to_fp(r) for r in rows}
        return [by_id.get(cid) for cid in comp_id_list]

    def get_fingerprint_settings(self) -> dict:
        """
        Return fingerprint settings used in the database.
        If multiple settings are found, return the first non-null set.
        """
        row = self._conn.execute(f"""
            SELECT fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype
            FROM {self.table}
            WHERE fp_nbits IS NOT NULL OR fp_radius IS NOT NULL
               OR fp_sparse IS NOT NULL OR fp_count IS NOT NULL OR fp_dtype IS NOT NULL
            LIMIT 1
        """).fetchone()
        if row:
            return {
                "nbits":  row["fp_nbits"],
                "radius": row["fp_radius"],
                "sparse": bool(row["fp_sparse"]) if row["fp_sparse"] is not None else None,
                "count":  bool(row["fp_count"])  if row["fp_count"]  is not None else None,
                "dtype":  row["fp_dtype"] or "float32",
            }
        return {
            "nbits": self.fingerprint_nbits,
            "radius": self.fingerprint_radius,
            "sparse": bool(self.fingerprint_sparse),
            "count": bool(self.fingerprint_count),
            "dtype": self.fingerprint_dtype_dense,
        }

    def get_compound(self, comp_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(f"""
            SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
            FROM {self.table}
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        return dict(row) if row else None

    def get_compounds(self, comp_ids: List[str]) -> pd.DataFrame:
        if not comp_ids:
            return pd.DataFrame(columns=[
                "comp_id", "smiles", "inchi", "inchikey", "classyfire_class", "classyfire_superclass"
            ])
        placeholders = ",".join("?" for _ in comp_ids)
        df = pd.read_sql_query(f"""
            SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
            FROM {self.table}
            WHERE comp_id IN ({placeholders})
        """, self._conn, params=comp_ids)
        if df.empty:
            return df
        order = {cid: i for i, cid in enumerate(comp_ids)}
        return (df
                .assign(__order=df["comp_id"].map(order))
                .sort_values("__order")
                .drop(columns="__order")
                .reset_index(drop=True))

    def sql_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self._conn)

    # ---------------- Compute fingerprints for missing rows ----------------

    def compute_fingerprints_missing(
        self,
        batch_size: int = 1000,
        use_progress_bar: bool = True,
        fp_size: Optional[int] = None,
        radius: Optional[int] = None,
        sparse: Optional[bool] = None,
        count: Optional[bool] = None,
    ) -> dict:
        """
        Compute fingerprints where all FP blobs are empty.
        Supports outputs from compute_morgan_fingerprints:
          - dense matrix (N, fp_size)
          - list of bit arrays (sparse/binary)
          - list of (bits, counts) (sparse/count)
        """
        radius = self.fingerprint_radius if radius is None else radius
        sparse = self.fingerprint_sparse if sparse is None else sparse
        count  = self.fingerprint_count  if count  is None else count
        fp_size = self.fingerprint_nbits if fp_size is None else fp_size

        def iter_missing(sql: str):
            offset = 0
            with self._conn as _:
                cur = self._conn.cursor()
                while True:
                    rows = cur.execute(sql, (batch_size, offset)).fetchall()
                    if not rows:
                        break
                    yield rows
                    offset += batch_size

        base_where = """
            COALESCE(LENGTH(fingerprint_bits),0)=0
            AND COALESCE(LENGTH(fingerprint_counts),0)=0
            AND COALESCE(LENGTH(fingerprint_dense),0)=0
        """
        sql_smiles = f"""
            SELECT comp_id, smiles FROM {self.table}
            WHERE smiles IS NOT NULL AND TRIM(smiles) <> '' AND {base_where}
            LIMIT ? OFFSET ?
        """
        sql_inchi = f"""
            SELECT comp_id, inchi FROM {self.table}
            WHERE (smiles IS NULL OR TRIM(smiles)='') AND inchi IS NOT NULL AND TRIM(inchi) <> '' AND {base_where}
            LIMIT ? OFFSET ?
        """

        stats = {"updated": 0, "attempted": 0, "skipped": 0}

        def _apply_dense(comp_ids: List[str], mat: np.ndarray):
            payloads = []
            for cid, rowvec in zip(comp_ids, mat):
                payloads.append((
                    encode_dense_fp(rowvec), b"", b"",  # dense, no sparse
                    fp_size, radius, 0, 1 if count else 0, self.fingerprint_dtype_dense, cid
                ))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_dense=?, fingerprint_bits=?, fingerprint_counts=?,
                            fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                        WHERE comp_id=?""",
                    payloads,
                )

        def _apply_sparse_bits(comp_ids: List[str], bitlists: List[np.ndarray]):
            payloads = []
            for cid, bits in zip(comp_ids, bitlists):
                b_blob, c_blob = encode_sparse_fp(bits, None)
                payloads.append((
                    b_blob, c_blob, b"",  # sparse bits, no dense
                    fp_size, radius, 1, 0, None, cid
                ))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_bits=?, fingerprint_counts=?, fingerprint_dense=?,
                            fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                        WHERE comp_id=?""",
                    payloads,
                )

        def _apply_sparse_counts(comp_ids: List[str], pairs: List[Tuple[np.ndarray, np.ndarray]]):
            payloads = []
            for cid, (bits, counts_arr) in zip(comp_ids, pairs):
                b_blob, c_blob = encode_sparse_fp(bits, counts_arr)
                payloads.append((
                    b_blob, c_blob, b"",  # sparse counts, no dense
                    fp_size, radius, 1, 1, None, cid
                ))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_bits=?, fingerprint_counts=?, fingerprint_dense=?,
                            fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                        WHERE comp_id=?""",
                    payloads,
                )

        for sql, which in ((sql_smiles, "smiles"), (sql_inchi, "inchi")):
            for rows in iter_missing(sql):
                comp_ids = [r[0] for r in rows]
                reps = [r[1] for r in rows]
                res = compute_morgan_fingerprints(
                    smiles=reps if which == "smiles" else None,
                    inchis=reps if which == "inchi" else None,
                    sparse=sparse, count=count, radius=radius,
                    progress_bar=use_progress_bar,
                )

                if isinstance(res, np.ndarray):
                    _apply_dense(comp_ids, res)
                    upd = res.shape[0]
                else:
                    if sparse and not count:
                        _apply_sparse_bits(comp_ids, res)       # type: ignore[arg-type]
                        upd = len(res)
                    elif sparse and count:
                        _apply_sparse_counts(comp_ids, res)     # type: ignore[arg-type]
                        upd = len(res)
                    else:
                        mat = np.vstack([np.asarray(x, dtype=np.float32) for x in res])
                        _apply_dense(comp_ids, mat)
                        upd = mat.shape[0]

                stats["updated"] += upd
                stats["attempted"] += len(comp_ids)

        # rows without SMILES & without InChI are skipped
        stats["skipped"] = self.sql_query(f"""
            SELECT COUNT(*) AS n
            FROM {self.table}
            WHERE (smiles IS NULL OR TRIM(smiles)='')
              AND (inchi  IS NULL OR TRIM(inchi) ='')
        """)["n"].iloc[0]

        return stats
