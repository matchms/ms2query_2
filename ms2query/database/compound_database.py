import sqlite3
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
SparseFP = Tuple[np.ndarray, Optional[np.ndarray]]  # (bits, counts) where counts may be None
AnyFP = Union[DenseFP, SparseFP]

@dataclass
class CompoundDatabase:
    """SQLite-based compound database with sparse fingerprint storage.
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
    # Default FP parameters (used by the backfill method)
    fingerprint_radius: int = 9
    fingerprint_sparse: bool = True
    fingerprint_count: bool = True
    fingerprint_nbits: int = 4096
    fingerprint_dtype_dense: str = "float32"  # for dense storage
    _conn: sqlite3.Connection = field(init=False, repr=False)

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

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.executescript(f"""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS {self.table}(
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
            CREATE INDEX IF NOT EXISTS idx_compounds_smiles ON {self.table}(smiles);
            CREATE INDEX IF NOT EXISTS idx_compounds_inchi  ON {self.table}(inchi);
        """)

        # Add any missing columns for existing DBs
        existing = {r[1] for r in cur.execute(f"PRAGMA table_info({self.table})").fetchall()}
        to_add = [
            ("smiles", "TEXT"),
            ("inchi", "TEXT"),
            ("inchikey", "TEXT"),
            ("fingerprint", "BLOB"),
            ("fingerprint_bits", "BLOB"),
            ("fingerprint_counts", "BLOB"),
            ("fingerprint_dense", "BLOB"),
            ("fp_nbits", "INTEGER"),
            ("fp_radius", "INTEGER"),
            ("fp_sparse", "INTEGER"),
            ("fp_count", "INTEGER"),
            ("fp_dtype", "TEXT"),
            ("classyfire_class", "TEXT"),
            ("classyfire_superclass", "TEXT"),
        ]
        for name, typ in to_add:
            if name not in existing:
                cur.execute(f"ALTER TABLE {self.table} ADD COLUMN {name} {typ}")

        # (Optional) best-effort: ensure indexes that don’t break old DBs
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_smiles ON {self.table}(smiles)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_inchi  ON {self.table}(inchi)")
        self._conn.commit()

    # ---------- UPSERTS ----------

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
        """Normalize caller-provided FP + parameters into DB columns."""
        cols = {
            "fingerprint_bits": b"",
            "fingerprint_counts": b"",
            "fingerprint_dense": b"",
            "fp_nbits": nbits if nbits is not None else self.fingerprint_nbits,
            "fp_radius": radius if radius is not None else self.fingerprint_radius,
            "fp_sparse": 1 if (self.fingerprint_sparse if sparse is None else sparse) else 0,
            "fp_count":  1 if (self.fingerprint_count  if count  is None else count)  else 0,
            "fp_dtype": dtype_dense or self.fingerprint_dtype_dense,
        }

        if fp is None:
            return cols

        # Decide on representation based on requested flags
        is_sparse = bool(cols["fp_sparse"])

        if is_sparse:
            # fp may be (bits, counts) or just bits
            if isinstance(fp, tuple):
                bits, counts = fp
                bits_blob, counts_blob = encode_sparse_fp(bits, counts)
            else:
                # bits only (binary)
                bits_blob, counts_blob = encode_sparse_fp(fp, None)  # type: ignore
            cols["fingerprint_bits"] = bits_blob
            cols["fingerprint_counts"] = counts_blob
            cols["fingerprint_dense"] = b""
        else:
            # Dense vector (binary or counts); force float32 on disk
            if isinstance(fp, tuple):
                raise ValueError("Dense fingerprint must be a single vector, not a (bits, counts) tuple.")
            cols["fingerprint_dense"] = encode_dense_fp(fp)  # type: ignore
            cols["fingerprint_bits"] = b""
            cols["fingerprint_counts"] = b""
        return cols

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
        if inchikey is None:
            raise ValueError("inchikey is required to form comp_id (inchikey14).")
        comp_id = inchikey14_from_full(inchikey)
        if not comp_id:
            raise ValueError(f"Invalid InChIKey: {inchikey}")

        cols = self._pack_fp_for_write(
            fingerprint,
            sparse=fp_sparse,
            count=fp_count,
            radius=fp_radius,
            nbits=fp_nbits,
            dtype_dense=fp_dtype,
        )

        cur = self._conn.cursor()
        cur.execute(f"""
            INSERT INTO {self.table} (
                comp_id, smiles, inchi, inchikey,
                fingerprint_bits, fingerprint_counts, fingerprint_dense,
                fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype,
                classyfire_class, classyfire_superclass
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(comp_id) DO UPDATE SET
                smiles=COALESCE(excluded.smiles, {self.table}.smiles),
                inchi=COALESCE(excluded.inchi, {self.table}.inchi),
                inchikey=COALESCE(excluded.inchikey, {self.table}.inchikey),

                fingerprint_bits=CASE
                    WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0) > 0
                    THEN excluded.fingerprint_bits ELSE {self.table}.fingerprint_bits END,
                fingerprint_counts=CASE
                    WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                    THEN excluded.fingerprint_counts ELSE {self.table}.fingerprint_counts END,
                fingerprint_dense=CASE
                    WHEN COALESCE(LENGTH(excluded.fingerprint_dense),0) > 0
                    THEN excluded.fingerprint_dense ELSE {self.table}.fingerprint_dense END,

                fp_nbits  = COALESCE(excluded.fp_nbits,  {self.table}.fp_nbits),
                fp_radius = COALESCE(excluded.fp_radius, {self.table}.fp_radius),
                fp_sparse = COALESCE(excluded.fp_sparse, {self.table}.fp_sparse),
                fp_count  = COALESCE(excluded.fp_count,  {self.table}.fp_count),
                fp_dtype  = COALESCE(excluded.fp_dtype,  {self.table}.fp_dtype),

                classyfire_class=COALESCE(excluded.classyfire_class, {self.table}.classyfire_class),
                classyfire_superclass=COALESCE(excluded.classyfire_superclass, {self.table}.classyfire_superclass)
        """, (
            comp_id, smiles, inchi, inchikey,
            cols["fingerprint_bits"], cols["fingerprint_counts"], cols["fingerprint_dense"],
            cols["fp_nbits"], cols["fp_radius"], cols["fp_sparse"], cols["fp_count"], cols["fp_dtype"],
            classyfire_class, classyfire_superclass,
        ))
        self._conn.commit()
        return comp_id

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> List[str]:
        comp_ids: List[str] = []
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            for r in rows:
                inchikey = r.get("inchikey")
                if not inchikey:
                    raise ValueError("Each row must contain 'inchikey'.")
                comp_id = inchikey14_from_full(inchikey)
                if not comp_id:
                    raise ValueError(f"Invalid InChIKey: {inchikey}")

                cols = self._pack_fp_for_write(
                    r.get("fingerprint"),
                    sparse=r.get("fp_sparse"),
                    count=r.get("fp_count"),
                    radius=r.get("fp_radius"),
                    nbits=r.get("fp_nbits"),
                    dtype_dense=r.get("fp_dtype"),
                )

                cur.execute(f"""
                    INSERT INTO {self.table} (
                        comp_id, smiles, inchi, inchikey,
                        fingerprint_bits, fingerprint_counts, fingerprint_dense,
                        fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype,
                        classyfire_class, classyfire_superclass
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(comp_id) DO UPDATE SET
                        smiles=COALESCE(excluded.smiles, {self.table}.smiles),
                        inchi=COALESCE(excluded.inchi, {self.table}.inchi),
                        inchikey=COALESCE(excluded.inchikey, {self.table}.inchikey),

                        fingerprint_bits=CASE
                            WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0) > 0
                            THEN excluded.fingerprint_bits ELSE {self.table}.fingerprint_bits END,
                        fingerprint_counts=CASE
                            WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                            THEN excluded.fingerprint_counts ELSE {self.table}.fingerprint_counts END,
                        fingerprint_dense=CASE
                            WHEN COALESCE(LENGTH(excluded.fingerprint_dense),0) > 0
                            THEN excluded.fingerprint_dense ELSE {self.table}.fingerprint_dense END,

                        fp_nbits = COALESCE(excluded.fp_nbits,  {self.table}.fp_nbits),
                        fp_radius = COALESCE(excluded.fp_radius, {self.table}.fp_radius),
                        fp_sparse = COALESCE(excluded.fp_sparse, {self.table}.fp_sparse),
                        fp_count = COALESCE(excluded.fp_count,  {self.table}.fp_count),
                        fp_dtype = COALESCE(excluded.fp_dtype,  {self.table}.fp_dtype),

                        classyfire_class=COALESCE(excluded.classyfire_class, {self.table}.classyfire_class),
                        classyfire_superclass=COALESCE(excluded.classyfire_superclass,
                          {self.table}.classyfire_superclass)
                """, (
                    comp_id,
                    r.get("smiles"),
                    r.get("inchi"),
                    inchikey,
                    cols["fingerprint_bits"], cols["fingerprint_counts"], cols["fingerprint_dense"],
                    cols["fp_nbits"], cols["fp_radius"], cols["fp_sparse"], cols["fp_count"], cols["fp_dtype"],
                    r.get("classyfire_class"),
                    r.get("classyfire_superclass"),
                ))
                comp_ids.append(comp_id)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return comp_ids

    def upsert_metadata_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        colmap: Optional[Dict[str, str]] = None,
        staging_table: str = "_staging_compounds",
    ) -> dict:
        """
        Ultra-simple loader for compound metadata (no fingerprints).
        Accepts a DataFrame that has at least an 'inchikey' column (can be remapped via `colmap`).
        Extra columns in the DataFrame are ignored.

        Steps:
        1) Build comp_id = inchikey14_from_full(inchikey)
        2) Keep only [comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass]
        3) Load into a temporary staging table via pandas.to_sql
        4) Single INSERT ... SELECT ... ON CONFLICT(comp_id) DO UPDATE to upsert

        Returns:
        {"rows": int, "valid": int, "inserted_or_updated": int, "skipped_no_or_bad_inchikey": int}
        """
        if df is None or df.empty:
            return {"rows": 0, "valid": 0, "inserted_or_updated": 0, "skipped_no_or_bad_inchikey": 0}

        # Map incoming columns -> our names (anything else is ignored)
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

        # Build a compact frame with just the columns we care about
        work = pd.DataFrame()
        work["inchikey"] = df[cmap["inchikey"]].astype(str)

        # Compute comp_id (inchikey14); drop invalid/missing
        work["comp_id"] = work["inchikey"].map(inchikey14_from_full)
        valid_mask = work["comp_id"].notna() & work["comp_id"].astype(str).str.len().eq(14)
        skipped = int((~valid_mask).sum())

        work = work.loc[valid_mask, ["comp_id", "inchikey"]].copy()

        # Optional columns (use .get to avoid KeyErrors)
        for k in ("smiles", "inchi", "classyfire_class", "classyfire_superclass"):
            src = cmap[k]
            work[k] = df[src] if src in df.columns else None

        # Deduplicate on comp_id, keeping the last occurrence
        work = work.drop_duplicates(subset=["comp_id"], keep="last").reset_index(drop=True)

        # Stage into SQLite (replace the staging table each call)
        work.to_sql(staging_table, self._conn, if_exists="replace", index=False)

        # Upsert from staging into compounds (fingerprint columns remain untouched/NULL)
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            # Use INSERT ... SELECT with ON CONFLICT(comp_id) DO UPDATE
            cur.execute(f"""
                INSERT INTO {self.table} (
                    comp_id, smiles, inchi, inchikey,
                    classyfire_class, classyfire_superclass
                )
                SELECT comp_id, smiles, inchi, inchikey, classyfire_class, classyfire_superclass
                FROM {staging_table}
                ON CONFLICT(comp_id) DO UPDATE SET
                    smiles=COALESCE(excluded.smiles, {self.table}.smiles),
                    inchi=COALESCE(excluded.inchi, {self.table}.inchi),
                    inchikey=COALESCE(excluded.inchikey, {self.table}.inchikey),
                    classyfire_class=COALESCE(excluded.classyfire_class, {self.table}.classyfire_class),
                    classyfire_superclass=COALESCE(excluded.classyfire_superclass, {self.table}.classyfire_superclass)
            """)
            affected = cur.rowcount if cur.rowcount is not None else 0
            # Clean up staging
            cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            # ensure staging dropped even on error
            try:
                cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
                self._conn.commit()
            except Exception:
                pass
            raise

        return {
            "rows": int(len(df)),
            "valid": int(len(work)),
            "inserted_or_updated": int(affected),
            "skipped_no_or_bad_inchikey": int(skipped),
        }

    # ---------- READ ----------

    def _row_to_fp(self, row: sqlite3.Row):
        """Convert one DB row with fingerprint blobs + metadata into the public return type."""
        dense_blob  = row["fingerprint_dense"] or b""
        bits_blob   = row["fingerprint_bits"] or b""
        counts_blob = row["fingerprint_counts"] or b""

        # Prefer dense if present
        if dense_blob:
            dtype = (row["fp_dtype"] or "float32")
            return decode_dense_fp(dense_blob, dtype=dtype)

        # Otherwise sparse
        if bits_blob or counts_blob:
            bits, counts = decode_sparse_fp(bits_blob, counts_blob)
            # If no counts were stored (binary), return bits array only
            if counts.size == 0:
                return bits
            return (bits, counts)

        # Nothing stored
        return None

    def get_fingerprint(self, comp_id: str):
        """
        Return this compound's fingerprint in a natural Python type:
        - dense: np.ndarray (float32, length fp_nbits)
        - sparse/binary: np.ndarray of uint32 bit indices
        - sparse/count: (np.ndarray[uint32] bits, np.ndarray[int32] counts)
        Returns None if no fingerprint is stored.
        """
        row = self._conn.execute(f"""
            SELECT fingerprint_bits, fingerprint_counts, fingerprint_dense,
                fp_nbits, fp_radius, fp_sparse, fp_count, fp_dtype
            FROM {self.table}
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        if not row:
            return None
        return self._row_to_fp(row)

    def get_fingerprints(self, comp_id_list: list[str]):
        """
        Batch version of get_fingerprint. Preserves input order.
        Missing comp_ids yield None in the corresponding slot.
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
        Return the FP settings used in this DB.
        If the table is empty (no stored metadata yet), fall back to the instance defaults.
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

        # Fallback: instance defaults (fresh DB, nothing computed yet)
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
        df["__order"] = df["comp_id"].map(order)
        df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        return df

    def sql_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self._conn)

    # ---------- Compute fingerprints later, for all missing ----------

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
        Compute fingerprints for all compounds with missing FPs.
        Adapts to all output variants from compute_morgan_fingerprints:
          - dense array of shape (N, fp_size)
          - list of np.ndarray (bits) for sparse/binary
          - list of (bits, counts) for sparse/count

        Stores into appropriate columns and sets fp_* metadata.
        """
        # parameters default to class defaults if not provided
        radius = self.fingerprint_radius if radius is None else radius
        sparse = self.fingerprint_sparse if sparse is None else sparse
        count  = self.fingerprint_count  if count  is None else count
        fp_size = self.fingerprint_nbits if fp_size is None else fp_size

        cur = self._conn.cursor()

        def _select_batch(sql: str, params: tuple) -> List[sqlite3.Row]:
            return cur.execute(sql, params).fetchall()

        def _update_rows_dense(comp_ids: List[str], mat: np.ndarray) -> int:
            updated = 0
            cur.execute("BEGIN")
            try:
                for cid, rowvec in zip(comp_ids, mat):
                    blob = encode_dense_fp(rowvec)
                    cur.execute(
                        f"""UPDATE {self.table}
                            SET fingerprint_dense=?,
                                fingerprint_bits=?,
                                fingerprint_counts=?,
                                fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                            WHERE comp_id=?""",
                        (blob, b"", b"", fp_size, radius, 0, 1 if count else 0, self.fingerprint_dtype_dense, cid)
                    )
                    updated += 1
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
            return updated

        def _update_rows_sparse_bits_only(comp_ids: List[str], bitlists: List[np.ndarray]) -> int:
            updated = 0
            cur.execute("BEGIN")
            try:
                for cid, bits in zip(comp_ids, bitlists):
                    b_blob, c_blob = encode_sparse_fp(bits, None)
                    cur.execute(
                        f"""UPDATE {self.table}
                            SET fingerprint_bits=?, fingerprint_counts=?,
                                fingerprint_dense=?,
                                fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                            WHERE comp_id=?""",
                        (b_blob, c_blob, b"", fp_size, radius, 1, 0, None, cid)
                    )
                    updated += 1
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
            return updated

        def _update_rows_sparse_with_counts(comp_ids: List[str], pairs: List[Tuple[np.ndarray, np.ndarray]]) -> int:
            updated = 0
            cur.execute("BEGIN")
            try:
                for cid, (bits, counts_arr) in zip(comp_ids, pairs):
                    b_blob, c_blob = encode_sparse_fp(bits, counts_arr)
                    cur.execute(
                        f"""UPDATE {self.table}
                            SET fingerprint_bits=?, fingerprint_counts=?,
                                fingerprint_dense=?,
                                fp_nbits=?, fp_radius=?, fp_sparse=?, fp_count=?, fp_dtype=?
                            WHERE comp_id=?""",
                        (b_blob, c_blob, b"", fp_size, radius, 1, 1, None, cid)
                    )
                    updated += 1
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
            return updated

        stats = {"updated": 0, "attempted": 0, "skipped": 0}

        # PASS A: SMILES-present & fingerprints missing
        sql_smiles = f"""
            SELECT comp_id, smiles
            FROM {self.table}
            WHERE smiles IS NOT NULL
              AND TRIM(smiles) <> ''
              AND COALESCE(LENGTH(fingerprint_bits),0)=0
              AND COALESCE(LENGTH(fingerprint_counts),0)=0
              AND COALESCE(LENGTH(fingerprint_dense),0)=0
            LIMIT ?
            OFFSET ?
        """

        # PASS B: no SMILES, but InChI-present & fingerprints missing
        sql_inchi = f"""
            SELECT comp_id, inchi
            FROM {self.table}
            WHERE (smiles IS NULL OR TRIM(smiles) = '')
              AND inchi IS NOT NULL
              AND TRIM(inchi) <> ''
              AND COALESCE(LENGTH(fingerprint_bits),0)=0
              AND COALESCE(LENGTH(fingerprint_counts),0)=0
              AND COALESCE(LENGTH(fingerprint_dense),0)=0
            LIMIT ?
            OFFSET ?
        """

        for sql, which in [(sql_smiles, "smiles"), (sql_inchi, "inchi")]:
            offset = 0
            while True:
                rows = _select_batch(sql, (batch_size, offset))
                if not rows:
                    break
                comp_ids = [r[0] for r in rows]
                reps = [r[1] for r in rows]  # SMILES or InChI strings

                # compute in one go
                res = compute_morgan_fingerprints(
                    smiles=reps if which == "smiles" else None,
                    inchis=reps if which == "inchi" else None,
                    sparse=sparse,
                    count=count,
                    radius=radius,
                    #n_bits=fp_size,
                    progress_bar=use_progress_bar,
                )

                # Detect shape/variant and write back
                if isinstance(res, np.ndarray):
                    # dense matrix (N, fp_size)
                    upd = _update_rows_dense(comp_ids, res)
                else:
                    # list-like results
                    if sparse and not count:
                        # list of np.ndarray of uint32 bit ids
                        upd = _update_rows_sparse_bits_only(comp_ids, res)  # type: ignore
                    elif sparse and count:
                        # list of (bits, counts)
                        upd = _update_rows_sparse_with_counts(comp_ids, res)  # type: ignore
                    else:
                        # Some implementations may still return a list of dense rows (rare).
                        # Normalize to matrix then write.
                        mat = np.vstack([np.asarray(x, dtype=np.float32) for x in res])
                        upd = _update_rows_dense(comp_ids, mat)

                stats["updated"] += upd
                stats["attempted"] += len(comp_ids)
                offset += batch_size

        # rows without SMILES & without InChI are skipped
        stats["skipped"] = self.sql_query(f"""
            SELECT COUNT(*) AS n
            FROM {self.table}
            WHERE (smiles IS NULL OR TRIM(smiles)='')
              AND (inchi  IS NULL OR TRIM(inchi) ='')
        """)["n"].iloc[0]

        return stats
