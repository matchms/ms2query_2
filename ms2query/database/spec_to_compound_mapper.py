import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
from ms2query.data_processing import inchikey14_from_full
from ms2query.database import CompoundDatabase


# ==================================================
# Mapping: spectrum <-> compound (spec_to_comp)
# ==================================================

@dataclass
class SpecToCompoundMap:
    """This class manages the mapping between spectrum IDs and compound IDs.
    
    Attributes
    ----------
    sqlite_path : str
        Path to the SQLite database file.
    table : str
        Database table name for the mapping (default: "spec_to_comp").
    compound_table : str
        Database table name for compounds (default: "compounds").
    """
    sqlite_path: str
    table: str = "spec_to_comp"
    compound_table: str = "compounds"

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
        # No strict FK enforcement (SpectralDatabase may have been created without FK pragma),
        # here: index both sides for fast lookup.
        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS {self.table}(
                spec_id TEXT NOT NULL,
                comp_id TEXT    NOT NULL,
                PRIMARY KEY (spec_id),
                CHECK (length(comp_id) = 14)
            );
            CREATE INDEX IF NOT EXISTS idx_spec_to_comp_comp ON {self.table}(comp_id);
        """)
        self._conn.commit()

    # ---------- API ----------

    def link(self, spec_id: str, comp_id: str):
        """Insert or replace a single mapping."""
        if not comp_id or len(comp_id) != 14:
            raise ValueError("comp_id must be inchikey14 (14 characters).")
        self._conn.execute(f"""
            INSERT INTO {self.table} (spec_id, comp_id)
            VALUES (?, ?)
            ON CONFLICT(spec_id) DO UPDATE SET comp_id=excluded.comp_id
        """, (spec_id, comp_id))
        self._conn.commit()

    def link_many(self, pairs: Iterable[Tuple[int, str]]):
        """Bulk link (spec_id, comp_id)."""
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.executemany(f"""
                INSERT INTO {self.table} (spec_id, comp_id)
                VALUES (?, ?)
                ON CONFLICT(spec_id) DO UPDATE SET comp_id=excluded.comp_id
            """, list(pairs))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # ---- getters: spec_id -> comp_id ----

    def get_comp_id_for_specs(self, spec_ids: List[str]) -> pd.DataFrame:
        """Return a DataFrame with columns [spec_id, comp_id] for the provided spec_ids."""
        if not spec_ids:
            return pd.DataFrame(columns=["spec_id", "comp_id"])
        placeholders = ",".join("?" * len(spec_ids))
        rows = self._conn.execute(
            f"SELECT spec_id, comp_id FROM {self.table} WHERE spec_id IN ({placeholders})",
            spec_ids
        ).fetchall()
        return pd.DataFrame(rows, columns=["spec_id", "comp_id"])

    def get_specs_for_comp(self, comp_id: str) -> List[str]:
        """Return list of spec_ids for a given comp_id."""
        rows = self._conn.execute(f"SELECT spec_id FROM {self.table} WHERE comp_id = ?", (comp_id,)).fetchall()
        return [r[0] for r in rows]

    def get_all_mappings(self) -> pd.DataFrame:
        """Return all spec_id <-> comp_id mappings as a DataFrame."""
        rows = self._conn.execute(f"SELECT spec_id, comp_id FROM {self.table}").fetchall()
        return pd.DataFrame(rows, columns=["spec_id", "comp_id"])


# ==================================================
# Integrations with SpectralDatabase
# ==================================================

def map_from_spectraldb_metadata(
    spectral_db_sqlite_path: str,
    mapping_sqlite_path: Optional[str] = None,
    compounds_sqlite_path: Optional[str] = None,
    spectra_table: str = "spectra",
    compound_table: str = "compounds",
    mapping_table: str = "spec_to_comp",
    *,
    create_missing_compounds: bool = True
) -> Tuple[int, int]:
    """
    Read spectra metadata (expects 'inchikey' in metadata), create comp_id (inchikey14),
    populate spec_to_comp, and optionally upsert minimal compounds.

    Returns: (n_mapped, n_new_compounds)
    """
    # We do not import the class to avoid circular imports; use sqlite directly.
    s_conn = sqlite3.connect(spectral_db_sqlite_path)
    s_conn.row_factory = sqlite3.Row

    map_db_path = mapping_sqlite_path or spectral_db_sqlite_path
    c_db_path = compounds_sqlite_path or spectral_db_sqlite_path

    mapper = SpecToCompoundMap(map_db_path, table=mapping_table)
    compdb = CompoundDatabase(c_db_path, table=compound_table)

    # Discover which columns exist in the spectra table
    cols = {r[1] for r in s_conn.execute(f"PRAGMA table_info({spectra_table})").fetchall()}
    want = ["spec_id", "inchikey", "smiles", "inchi", "classyfire_class", "classyfire_superclass"]
    have = [c for c in want if c in cols]
    select_cols = ", ".join(have)

    rows = s_conn.execute(f"SELECT {select_cols} FROM {spectra_table}").fetchall()

    to_link: List[Tuple[int, str]] = []
    new_comp_rows: List[Dict[str, Any]] = []

    for r in rows:
        r = dict(r)
        spec_id = r["spec_id"]
        ik_full = r.get("inchikey")
        if not ik_full:
            continue
        comp_id = inchikey14_from_full(ik_full)
        if not comp_id:
            continue
        to_link.append((spec_id, comp_id))

        if create_missing_compounds:
            new_comp_rows.append({
                "smiles": r.get("smiles"),
                "inchi": r.get("inchi"),
                "inchikey": ik_full,
                "classyfire_class": r.get("classyfire_class"),
                "classyfire_superclass": r.get("classyfire_superclass"),
                "fingerprint": None,  # backfill later
            })

    # Bulk linking
    if to_link:
        mapper.link_many(to_link)

    # Upsert compounds
    n_new_compounds = 0
    if create_missing_compounds and new_comp_rows:
        # Deduplicate by comp_id to avoid redundant upserts
        seen: set[str] = set()
        dedup_rows: List[Dict[str, Any]] = []
        for r in new_comp_rows:
            cid = inchikey14_from_full(r["inchikey"])
            if cid and cid not in seen:
                seen.add(cid)
                dedup_rows.append(r)
        before = compdb.sql_query(f"SELECT COUNT(*) AS n FROM {compound_table}")["n"].iloc[0]
        compdb.upsert_many(dedup_rows)
        after  = compdb.sql_query(f"SELECT COUNT(*) AS n FROM {compound_table}")["n"].iloc[0]
        n_new_compounds = int(after - before)

    n_mapped = len(to_link)

    # Close connections
    mapper.close()
    compdb.close()
    s_conn.close()

    return n_mapped, n_new_compounds


def get_unique_compounds_from_spectraldb(
    spectral_db_sqlite_path: str,
    spectra_table: str = "spectra",
    external_meta: Optional[pd.DataFrame] = None,
    external_key_col: str = "inchikey14"
) -> pd.DataFrame:
    """
    Return a DataFrame of unique compounds present in the spectral DB, inferred via inchikey → inchikey14.
    Columns: inchikey14, inchikey (full), n_spectra. If `external_meta` is provided,
    it will be left-joined on `external_key_col` (default 'inchikey14').
    """
    conn = sqlite3.connect(spectral_db_sqlite_path)
    conn.row_factory = sqlite3.Row

    # pull spec_id + inchikey from spectra
    df = pd.read_sql_query(f"SELECT spec_id, inchikey FROM {spectra_table}", conn)
    conn.close()

    if df.empty:
        base = pd.DataFrame(columns=["inchikey14", "inchikey", "n_spectra"])
        if external_meta is not None:
            return base.merge(external_meta, how="left", left_on="inchikey14", right_on=external_key_col)
        return base

    # Compute inchikey14
    ik14 = df["inchikey"].fillna("").map(inchikey14_from_full)
    df["inchikey14"] = ik14

    # Aggregate
    agg = (
        df.dropna(subset=["inchikey14"])
          .groupby(["inchikey14"], as_index=False)
          .agg(n_spectra=("spec_id", "count"),
               inchikey=("inchikey", "first"))  # first full key seen
    )

    # Optional join with external meta
    if external_meta is not None and not external_meta.empty:
        agg = agg.merge(external_meta, how="left", left_on="inchikey14", right_on=external_key_col)

    # Order by prevalence
    agg = agg.sort_values("n_spectra", ascending=False).reset_index(drop=True)
    return agg
