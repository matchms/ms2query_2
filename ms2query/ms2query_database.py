import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from ms2query.data_processing import inchikey14_from_full
from ms2query.database import (
    CompoundDatabase,
    SpecToCompoundMap,
    SpectralDatabase,
    map_from_spectraldb_metadata,
)


# ================================ public wrapper ==============================

@dataclass
class MS2QueryDatabase:
    """Wrapper class as main hub/glue between the different MS2Query database elements.

    Responsibilities
    ----------------
    * Own a single SQLite path and initialize component tables if needed.
    * Provide one-stop creation from (already processed!) `matchms.Spectrum` objects.
    * Offer ergonomic retrievals by `spec_id`, `comp_id` (inchikey14).
    * Keep *types and table access paths* in one place.
    
    """

    sqlite_path: str
    ref_spectra_table: str = "spectra"
    ref_compound_table: str = "compounds"
    non_annotated_compound_table: str = "compounds_all"
    metadata_fields: List[str] = field(default_factory=lambda: [
        "precursor_mz", "ionmode", "smiles", "inchikey", "inchi", "name",
        "charge", "instrument_type", "adduct", "collision_energy"
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
        self.all_cdb = CompoundDatabase(self.sqlite_path, table=self.non_annotated_compound_table)
        self.mapper = SpecToCompoundMap(self.sqlite_path, compound_table=self.ref_compound_table)


    # ----------------------------- creation pipeline -----------------------------

    def create_from_spectra(
        self,
        spectra: List[Any],  # matchms.Spectrum
        *,
        map_compounds: bool = True,
        create_missing_compounds: bool = True,
    ) -> Dict[str, int]:
        """Ingest spectra -> (optionally) create spec↔comp links & upsert compounds.

        Parameters
        ----------
        spectra: List[matchms.Spectrum]
            List of matchms Spectrum objects to be inserted into the database.
        map_compounds: bool, default=True
            Whether to map spectra to compounds based on metadata InChIKeys.
        create_missing_compounds: bool, default=True
            Whether to create compound entries for spectra that do not have a matching compound yet.

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
        return {
            "n_inserted_spectra": len(spec_ids),
            "n_mapped": int(n_mapped),
            "n_new_compounds": int(n_new),
        }
    
    def add_second_compound_database(self, df):
        """Add an additional 'all compound' database without need for spectral data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing inchikey and other relevant compound information.
            Should at least contain smiles or inchi.
        """
        self.all_cdb.upsert_metadata_from_dataframe(df)

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
