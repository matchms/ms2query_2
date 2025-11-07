import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ms2query.data_processing import inchikey14_from_full
from ms2query.database import (
    CompoundDatabase,
    SpecToCompoundMap,
    SpectralDatabase,
    blob_to_array,
    ensure_merged_tables,
    map_from_spectraldb_metadata,
)


# ================================ public wrapper ==============================

@dataclass
class MS2QueryDatabase:
    """Thin facade around the 3 SQLite-backed components + merged tables.

    Responsibilities
    ----------------
    * Own a single SQLite path and initialize component tables if needed.
    * Provide one-stop creation from raw `matchms.Spectrum` objects.
    * Offer ergonomic retrievals by `spec_id`, `comp_id` (inchikey14), and `merged_id`.
    * Keep *types and table access paths* in one place.

    Notes
    -----
    - This wrapper uses separate connections created by the component classes.
      If strict single-transaction semantics across components is required,
      consider extending the components to accept an injected `sqlite3.Connection`.
    """

    sqlite_path: str
    ref_spectra_table: str = "spectra"
    ref_compound_table: str = "compounds"
    non_annotated_compound_table: str = "compounds_all"
    metadata_fields: List[str] = field(default_factory=lambda: [
        "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
        "instrument_type", "adduct", "collision_energy"
    ])

    # component singletons
    ref_sdb: SpectralDatabase = field(init=False)
    ref_cdb: CompoundDatabase = field(init=False)
    all_cdb: CompoundDatabase = field(init=False)
    mapper: SpecToCompoundMap = field(init=False)

    def __post_init__(self):
        # Initialize components (each manages its own connection)
        self.ref_sdb = SpectralDatabase(self.sqlite_path, table=self.ref_spectra_table,
                                    metadata_fields=self.metadata_fields)
        self.ref_cdb = CompoundDatabase(self.sqlite_path, table=self.ref_compound_table)
        self.all_cfb = CompoundDatabase(self.sqlite_path, table=self.non_annotated_compound_table)
        self.mapper = SpecToCompoundMap(self.sqlite_path, compound_table=self.ref_compound_table)
        # Ensure merged tables exist on the *same* file
        with sqlite3.connect(self.sqlite_path) as conn:
            ensure_merged_tables(conn)

    # ----------------------------- creation pipeline -----------------------------

    def create_from_spectra(
        self,
        spectra: List[Any],  # matchms.Spectrum
        *,
        map_compounds: bool = True,
        create_missing_compounds: bool = True,
    ) -> Dict[str, int]:
        """Ingest spectra -> (optionally) create spec↔comp links & upsert compounds.

        Returns counts: {"n_inserted_spectra": int, "n_mapped": int, "n_new_compounds": int}
        """
        spec_ids = self.ref_sdb.add_spectra(spectra)
        n_mapped = 0
        n_new = 0
        if map_compounds and spec_ids:
            n_mapped, n_new = map_from_spectraldb_metadata(
                spectral_db_sqlite_path=self.sqlite_path,
                mapping_sqlite_path=self.sqlite_path,
                compounds_sqlite_path=self.sqlite_path,
                spectra_table=self.ref_spectra_table,
                compound_table=self.ref_compound_table,
                mapping_table="spec_to_comp",
                create_missing_compounds=create_missing_compounds,
            )
            # ONLY PLACEHOLDER --> LATER: ADD COMPOUNDS FROM LIST/FILE
            _, _ = map_from_spectraldb_metadata(
                spectral_db_sqlite_path=self.sqlite_path,
                mapping_sqlite_path=self.sqlite_path,
                compounds_sqlite_path=self.sqlite_path,
                spectra_table=self.ref_spectra_table,
                compound_table=self.non_annotated_compound_table,
                mapping_table="spec_to_comp_all",
                create_missing_compounds=create_missing_compounds,
            )
        return {
            "n_inserted_spectra": len(spec_ids),
            "n_mapped": int(n_mapped),
            "n_new_compounds": int(n_new),
        }

    # --------------------------------- retrievals --------------------------------
    # ---- by spec_id ----

    def spectra_by_spec_ids(self, spec_ids: List[int]):
        return self.ref_sdb.get_spectra_by_ids(spec_ids)

    def fragments_by_spec_ids(self, spec_ids: List[int]):
        return self.ref_sdb.get_fragments_by_ids(spec_ids)

    def metadata_by_spec_ids(self, spec_ids: List[int]) -> pd.DataFrame:
        return self.ref_sdb.get_metadata_by_ids(spec_ids)

    # ---- by comp_id (inchikey14) ----

    def spec_ids_by_comp_id(self, comp_id: str) -> List[int]:
        return self.mapper.get_specs_for_comp(comp_id)

    def spectra_by_comp_id(self, comp_id: str):
        return self.ref_sdb.get_spectra_by_ids(self.spec_ids_by_comp_id(comp_id))

    def metadata_by_comp_id(self, comp_id: str) -> pd.DataFrame:
        spec_ids = self.spec_ids_by_comp_id(comp_id)
        return self.ref_sdb.get_metadata_by_ids(spec_ids)

    def compound(self, comp_id: str) -> Optional[Dict[str, Any]]:
        return self.ref_cdb.get_compound(comp_id)

    # ---- merged spectra ----

    def merged_rows_by_comp_id(self, comp_id: str) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM merged_spectra WHERE comp_id = ? ORDER BY merged_id",
                conn,
                params=(comp_id,),
            )
        return df

    def merged_row(self, merged_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT * FROM merged_spectra WHERE merged_id = ?", (merged_id,)).fetchone()
            return dict(row) if row else None

    def merged_spectrum_arrays(self, merged_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r = self.merged_row(merged_id)
        if not r:
            return None
        # By convention in `spectra_merging.py`: mz=float64, intensities=float32
        mz = blob_to_array(r["mz"], np.float64)
        it = blob_to_array(r["intensities"], np.float32)
        return mz, it

    # -------------------------------- convenience SQL ------------------------------

    def sql(self, query: str) -> pd.DataFrame:
        """Run a read-only SQL query on the shared SQLite file."""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            return pd.read_sql_query(query, conn)

    # ----------------------------------- utilities ---------------------------------

    def inchikey_to_comp_id(self, inchikey_full: str) -> Optional[str]:
        return inchikey14_from_full(inchikey_full)

    def close(self):
        # Close component connections
        try:
            self.ref_sdb.close()
        except Exception:
            pass
        try:
            self.ref_cdb.close()
        except Exception:
            pass
        try:
            self.mapper.close()
        except Exception:
            pass
