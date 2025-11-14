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

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS {table}(
    comp_id               TEXT PRIMARY KEY,          -- inchikey14
    smiles                TEXT,
    inchi                 TEXT,
    inchikey              TEXT UNIQUE,

    -- sparse storage
    fingerprint_bits      BLOB,                      -- uint32[]
    fingerprint_counts    BLOB,                      -- int32[] (empty if binary)

    -- dense storage
    fingerprint_dense     BLOB,                      -- float32[]

    classyfire_class      TEXT,
    classyfire_superclass TEXT
);
CREATE INDEX IF NOT EXISTS idx_{table}_smiles ON {table}(smiles);
CREATE INDEX IF NOT EXISTS idx_{table}_inchi  ON {table}(inchi);
"""

SETTINGS_SQL = """
CREATE TABLE IF NOT EXISTS {settings_table}(
    id     INTEGER PRIMARY KEY CHECK (id=1),
    nbits  INTEGER NOT NULL,
    radius INTEGER NOT NULL,
    sparse INTEGER NOT NULL,   -- 1/0
    count  INTEGER NOT NULL,   -- 1/0
    dtype  TEXT    NOT NULL
);
"""

UPSERT_SQL = """
INSERT INTO {table} (
    comp_id, smiles, inchi, inchikey,
    fingerprint_bits, fingerprint_counts, fingerprint_dense,
    classyfire_class, classyfire_superclass
) VALUES (?,?,?,?,?,?,?, ?, ?)
ON CONFLICT(comp_id) DO UPDATE SET
    smiles                = COALESCE(excluded.smiles,                {table}.smiles),
    inchi                 = COALESCE(excluded.inchi,                 {table}.inchi),
    inchikey              = COALESCE(excluded.inchikey,              {table}.inchikey),
    fingerprint_bits      = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_bits),0)   > 0
                                 THEN excluded.fingerprint_bits ELSE {table}.fingerprint_bits END,
    fingerprint_counts    = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_counts),0) > 0
                                 THEN excluded.fingerprint_counts ELSE {table}.fingerprint_counts END,
    fingerprint_dense     = CASE WHEN COALESCE(LENGTH(excluded.fingerprint_dense),0)  > 0
                                 THEN excluded.fingerprint_dense ELSE {table}.fingerprint_dense END,
    classyfire_class      = COALESCE(excluded.classyfire_class,      {table}.classyfire_class),
    classyfire_superclass = COALESCE(excluded.classyfire_superclass, {table}.classyfire_superclass)
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
    # ---- SINGLE SOURCE OF TRUTH for FP settings ---
    fingerprint_radius: int = 9
    fingerprint_sparse: bool = True
    fingerprint_count: bool = True
    fingerprint_nbits: int = 4096
    fingerprint_dtype_dense: str = "float32"

    _conn: sqlite3.Connection = field(init=False, repr=False)

    # ---------------- lifecycle ----------------

    def __post_init__(self):
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema_and_settings()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def settings_table(self) -> str:
        return f"{self.table}_settings"

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

    def _ensure_schema_and_settings(self):
        """Create data table + settings table; sync instance settings with DB row."""
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL.format(table=self.table))
        cur.executescript(SETTINGS_SQL.format(settings_table=self.settings_table))
        # Sync: if row exists -> adopt it; else write our instance defaults.
        row = cur.execute(f"SELECT nbits, radius, sparse, count, dtype FROM {self.settings_table} WHERE id=1").fetchone()
        if row:
            # Adopt stored settings into the instance to guarantee consistency across sessions
            self.fingerprint_nbits = int(row["nbits"])
            self.fingerprint_radius = int(row["radius"])
            self.fingerprint_sparse = bool(row["sparse"])
            self.fingerprint_count = bool(row["count"])
            self.fingerprint_dtype_dense = str(row["dtype"])
        else:
            cur.execute(
                f"INSERT INTO {self.settings_table}(id, nbits, radius, sparse, count, dtype) VALUES (1,?,?,?,?,?)",
                (self.fingerprint_nbits, self.fingerprint_radius, int(self.fingerprint_sparse),
                 int(self.fingerprint_count), self.fingerprint_dtype_dense),
            )
        self._conn.commit()

# ---------------- settings API ----------------

    def _fingerprints_exist(self) -> bool:
        """Return True if any row has a non-empty fingerprint blob."""
        row = self._conn.execute(f"""
            SELECT 1 FROM {self.table}
            WHERE COALESCE(LENGTH(fingerprint_bits),0)   > 0
               OR COALESCE(LENGTH(fingerprint_counts),0) > 0
               OR COALESCE(LENGTH(fingerprint_dense),0)  > 0
            LIMIT 1
        """).fetchone()
        return bool(row)

    def set_fingerprint_parameters(
        self,
        *,
        radius: Optional[int] = None,
        sparse: Optional[bool] = None,
        count: Optional[bool] = None,
        nbits: Optional[int] = None,
        dtype_dense: Optional[str] = None,
    ) -> None:
        """
        Update the instance's FP parameters, but ONLY if no fingerprints are present.
        """
        if self._fingerprints_exist():
            raise RuntimeError(
                "Cannot change fingerprint parameters: fingerprints already exist in the database. "
                "Create a new DB or clear fingerprint columns first."
            )
        if radius is not None:
            self.fingerprint_radius = int(radius)
        if sparse is not None:
            self.fingerprint_sparse = bool(sparse)
        if count is not None: 
            self.fingerprint_count  = bool(count)
        if nbits is not None:
            self.fingerprint_nbits  = int(nbits)
        if dtype_dense is not None:
            self.fingerprint_dtype_dense = str(dtype_dense)

        with self._tx() as cur:
            cur.execute(
                f"""INSERT INTO {self.settings_table}(id, nbits, radius, sparse, count, dtype)
                    VALUES (1,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        nbits=excluded.nbits,
                        radius=excluded.radius,
                        sparse=excluded.sparse,
                        count=excluded.count,
                        dtype=excluded.dtype""",
                (self.fingerprint_nbits, self.fingerprint_radius, int(self.fingerprint_sparse),
                 int(self.fingerprint_count), self.fingerprint_dtype_dense),
            )

    # ---------------- helpers ----------------

    def _pack_fp_for_write(self, fp: Optional[AnyFP]) -> Dict[str, Any]:
        """Map a fingerprint to column payload using current instance settings."""
        if fp is None:
            return {"fingerprint_bits": b"", "fingerprint_counts": b"", "fingerprint_dense": b""}

        if self.fingerprint_sparse:
            if isinstance(fp, tuple):
                bits, counts = fp
                b_blob, c_blob = encode_sparse_fp(bits, counts)
            else:
                b_blob, c_blob = encode_sparse_fp(fp, None)  # type: ignore[arg-type]
            return {"fingerprint_bits": b_blob, "fingerprint_counts": c_blob, "fingerprint_dense": b""}
        else:
            if isinstance(fp, tuple):
                raise ValueError("Dense fingerprint must be a single vector, not (bits, counts).")
            return {"fingerprint_bits": b"", "fingerprint_counts": b"", "fingerprint_dense": encode_dense_fp(fp)}  # type: ignore[arg-type]

    def _row_to_fp(self, row: sqlite3.Row):
        dense_blob  = row["fingerprint_dense"]  or b""
        bits_blob   = row["fingerprint_bits"]   or b""
        counts_blob = row["fingerprint_counts"] or b""
        if dense_blob:
            return decode_dense_fp(dense_blob, dtype=self.fingerprint_dtype_dense)
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
    ) -> str:
        if not inchikey:
            raise ValueError("inchikey is required to form comp_id (inchikey14).")
        comp_id = inchikey14_from_full(inchikey)
        if not comp_id:
            raise ValueError(f"Invalid InChIKey: {inchikey}")

        cols = self._pack_fp_for_write(fingerprint)
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

            cols = self._pack_fp_for_write(r.get("fingerprint"))
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

    def compute_fingerprints(
        self,
        *,
        smiles: Optional[List[str]] = None,
        inchis: Optional[List[str]] = None,
        progress_bar: bool = False,
    ):
        """
        Compute Morgan fingerprints for the provided molecules using THIS INSTANCE'S settings.
        Does NOT write to the database.
        Provide exactly one of (smiles, inchis).

        Returns the same shapes/types as compute_morgan_fingerprints:
          - dense: np.ndarray of shape (N, nbits)
          - sparse/binary: List[np.ndarray[uint32]]
          - sparse/count:  List[Tuple[np.ndarray[uint32], np.ndarray[int32]]]
        """
        if (smiles is None) == (inchis is None):
            raise ValueError("Provide exactly one of 'smiles' or 'inchis'.")

        return compute_morgan_fingerprints(
            smiles=smiles,
            inchis=inchis,
            sparse=self.fingerprint_sparse,
            count=self.fingerprint_count,
            radius=self.fingerprint_radius,
            n_bits=self.fingerprint_nbits,
            progress_bar=progress_bar,
        )

    # ---------------- READ ----------------

    def get_fingerprint(self, comp_id: str):
        row = self._conn.execute(f"""
            SELECT fingerprint_bits, fingerprint_counts, fingerprint_dense
            FROM {self.table}
            WHERE comp_id = ?
        """, (comp_id,)).fetchone()
        if not row:
            return None
        dense_blob  = row["fingerprint_dense"]  or b""
        bits_blob   = row["fingerprint_bits"]   or b""
        counts_blob = row["fingerprint_counts"] or b""
        if dense_blob:
            return decode_dense_fp(dense_blob, dtype=self.fingerprint_dtype_dense)
        if bits_blob or counts_blob:
            bits, counts = decode_sparse_fp(bits_blob, counts_blob)
            return bits if counts.size == 0 else (bits, counts)
        return None

    def get_fingerprints(self, comp_id_list: List[str]):
        if not comp_id_list:
            return []
        placeholders = ",".join("?" for _ in comp_id_list)
        rows = self._conn.execute(f"""
            SELECT comp_id, fingerprint_bits, fingerprint_counts, fingerprint_dense
            FROM {self.table}
            WHERE comp_id IN ({placeholders})
        """, comp_id_list).fetchall()
        out = []
        for cid in comp_id_list:
            r = next((row for row in rows if row["comp_id"] == cid), None)
            if r is None:
                out.append(None); continue
            dense_blob  = r["fingerprint_dense"] or b""
            bits_blob   = r["fingerprint_bits"] or b""
            counts_blob = r["fingerprint_counts"] or b""
            if dense_blob:
                out.append(decode_dense_fp(dense_blob, dtype=self.fingerprint_dtype_dense))
            elif bits_blob or counts_blob:
                bits, counts = decode_sparse_fp(bits_blob, counts_blob)
                out.append(bits if counts.size == 0 else (bits, counts))
            else:
                out.append(None)
        return out

    def get_fingerprint_settings(self) -> dict:
        """Return the instance-level FP settings (authoritative)."""
        return {
            "nbits":  self.fingerprint_nbits,
            "radius": self.fingerprint_radius,
            "sparse": bool(self.fingerprint_sparse),
            "count":  bool(self.fingerprint_count),
            "dtype":  self.fingerprint_dtype_dense,
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
    ) -> dict:
        """Compute fingerprints for rows lacking any, using instance settings only."""
        fp_size = self.fingerprint_nbits
        radius  = self.fingerprint_radius
        sparse  = self.fingerprint_sparse
        count   = self.fingerprint_count

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
                payloads.append((encode_dense_fp(rowvec), b"", b"", cid))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_dense=?, fingerprint_bits=?, fingerprint_counts=?
                        WHERE comp_id=?""",
                    payloads,
                )

        def _apply_sparse_bits(comp_ids: List[str], bitlists: List[np.ndarray]):
            payloads = []
            for cid, bits in zip(comp_ids, bitlists):
                b_blob, c_blob = encode_sparse_fp(bits, None)
                payloads.append((b_blob, c_blob, b"", cid))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_bits=?, fingerprint_counts=?, fingerprint_dense=?
                        WHERE comp_id=?""",
                    payloads,
                )

        def _apply_sparse_counts(comp_ids: List[str], pairs: List[Tuple[np.ndarray, np.ndarray]]):
            payloads = []
            for cid, (bits, counts_arr) in zip(comp_ids, pairs):
                b_blob, c_blob = encode_sparse_fp(bits, counts_arr)
                payloads.append((b_blob, c_blob, b"", cid))
            with self._tx() as cur:
                cur.executemany(
                    f"""UPDATE {self.table}
                        SET fingerprint_bits=?, fingerprint_counts=?, fingerprint_dense=?
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
                    n_bits=fp_size,
                    progress_bar=use_progress_bar,
                )

                if isinstance(res, np.ndarray):                # dense
                    _apply_dense(comp_ids, res)
                    upd = res.shape[0]
                else:
                    if sparse and not count:                   # sparse/binary
                        _apply_sparse_bits(comp_ids, res)       # type: ignore[arg-type]
                        upd = len(res)
                    elif sparse and count:                     # sparse/count
                        _apply_sparse_counts(comp_ids, res)     # type: ignore[arg-type]
                        upd = len(res)
                    else:                                      # defensive: list of dense rows
                        mat = np.vstack([np.asarray(x, dtype=np.float32) for x in res])
                        _apply_dense(comp_ids, mat)
                        upd = mat.shape[0]

                stats["updated"] += upd
                stats["attempted"] += len(comp_ids)

        stats["skipped"] = self.sql_query(f"""
            SELECT COUNT(*) AS n
            FROM {self.table}
            WHERE (smiles IS NULL OR TRIM(smiles)='')
              AND (inchi  IS NULL OR TRIM(inchi) ='')
        """)["n"].iloc[0]

        return stats
