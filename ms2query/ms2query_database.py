import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
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
    metadata_fields: List[str] = field(
        default_factory=lambda: [
            "precursor_mz",
            "ionmode",
            "smiles",
            "inchikey",
            "inchi",
            "name",
            "charge",
            "instrument_type",
            "adduct",
            "collision_energy",
        ]
    )

    # component singletons
    ref_sdb: SpectralDatabase = field(init=False)
    ref_cdb: CompoundDatabase = field(init=False)
    all_cdb: CompoundDatabase = field(init=False)
    mapper: SpecToCompoundMap = field(init=False)

    def __post_init__(self):
        # Initialize components (each manages its own connection)
        self.ref_sdb = SpectralDatabase(
            self.sqlite_path,
            table=self.ref_spectra_table,
            metadata_fields=self.metadata_fields,
        )
        self.ref_cdb = CompoundDatabase(self.sqlite_path, table=self.ref_compound_table)
        self.all_cdb = CompoundDatabase(
            self.sqlite_path, table=self.non_annotated_compound_table
        )
        self.mapper = SpecToCompoundMap(
            self.sqlite_path, compound_table=self.ref_compound_table
        )

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
        spectra : List[matchms.Spectrum]
            List of matchms Spectrum objects to be inserted into the database.
        map_compounds : bool, default=True
            Whether to map spectra to compounds based on metadata InChIKeys.
        create_missing_compounds : bool, default=True
            Whether to create compound entries for spectra that do not have a matching compound yet.

        Returns
        -------
        dict
            Counts: {"n_inserted_spectra": int, "n_mapped": int, "n_new_compounds": int}
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

    def add_second_compound_database(self, df: pd.DataFrame) -> None:
        """Add an additional 'all compound' database without need for spectral data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing inchikey and other relevant compound information.
            Should at least contain smiles or inchi.
        """
        self.all_cdb.overwrite_metadata_from_dataframe(df)

    # --------------------------------- retrievals --------------------------------
    # ---- by spec_id ----

    def spectra_by_spec_ids(self, spec_ids: Sequence[str]):
        """Return list[Spectrum] for the given spec_ids."""
        return self.ref_sdb.get_spectra_by_ids(list(spec_ids))

    def fragments_by_spec_ids(self, spec_ids: Sequence[str]):
        """Return list[(mz, intensity)] for the given spec_ids."""
        return self.ref_sdb.get_fragments_by_ids(list(spec_ids))

    def metadata_by_spec_ids(self, spec_ids: Sequence[str]) -> pd.DataFrame:
        """Return metadata DataFrame for the given spec_ids."""
        return self.ref_sdb.get_metadata_by_ids(list(spec_ids))

    def embeddings_by_spec_ids(self, spec_ids: Sequence[str]):
        """Return (ids, embeddings) tuple for the given spec_ids."""
        return self.ref_sdb.get_embeddings(spec_ids=list(spec_ids))

    # ---- by comp_ids (inchikey14) ----

    def spec_ids_by_comp_ids(self, comp_ids: Sequence[str]) -> pd.DataFrame:
        """
        Return mapping of comp_ids -> spec_ids.

        Returns
        -------
        pd.DataFrame
            Columns: ['comp_id', 'spec_id'].
            One row per existing mapping (1:N).
        """
        return self.mapper.get_specs_for_comps(list(comp_ids))

    def spectra_by_comp_ids(self, comp_ids: Sequence[str]):
        """
        Return all spectra mapped to any of the given comp_ids.

        Notes
        -----
        * The order of spectra is determined by the underlying `get_spectra_by_ids`
          implementation and mapping table.
        * If you need to know which comp_id each spectrum belongs to, combine this
          with `spec_ids_by_comp_ids`.
        """
        df_map = self.mapper.get_specs_for_comps(list(comp_ids))
        if df_map.empty:
            return []
        spec_ids = df_map["spec_id"].tolist()
        return self.ref_sdb.get_spectra_by_ids(spec_ids)

    def metadata_by_comp_ids(self, comp_ids: Sequence[str]) -> pd.DataFrame:
        """
        Return metadata for all spectra mapped to the given comp_ids.

        Returns
        -------
        pd.DataFrame
            Columns: ['comp_id', 'spec_id', ...metadata_fields...]
        """
        df_map = self.mapper.get_specs_for_comps(list(comp_ids))
        if df_map.empty:
            cols = ["comp_id", "spec_id"] + self.metadata_fields
            return pd.DataFrame(columns=cols)

        meta = self.ref_sdb.get_metadata_by_ids(df_map["spec_id"].tolist())
        # meta: spec_id + metadata_fields
        out = df_map.merge(meta, on="spec_id", how="inner")
        # Ensure column order: comp_id, spec_id, metadata...
        return out[["comp_id", "spec_id"] + self.metadata_fields]

    def embeddings_by_comp_ids(self, comp_ids: Sequence[str]):
        """
        Return embeddings for all spectra mapped to the given comp_ids.

        Returns
        -------
        (np.ndarray, np.ndarray)
            (spec_ids, embeddings) as returned by SpectralDatabase.get_embeddings.
        """
        df_map = self.mapper.get_specs_for_comps(list(comp_ids))
        if df_map.empty:
            # Mirror SpectralDatabase.get_embeddings empty contract
            return (
                np.empty((0,), dtype=str),
                np.empty((0, 0), dtype=np.float32),
            )
        spec_ids = df_map["spec_id"].tolist()
        return self.ref_sdb.get_embeddings(spec_ids=spec_ids)

    # -------------------------------- convenience SQL ------------------------------

    def sql(self, query: str) -> pd.DataFrame:
        """Run a read-only SQL query on the shared SQLite file."""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            return pd.read_sql_query(query, conn)

    # ----------------------------------- utilities ---------------------------------

    def inchikey_to_comp_id(self, inchikey_full: str) -> Optional[str]:
        """Convert a full InChIKey to the 14-character comp_id (inchikey14)."""
        return inchikey14_from_full(inchikey_full)

    def close(self) -> None:
        """Close all component connections."""
        for obj in (self.ref_sdb, self.ref_cdb, self.all_cdb, self.mapper):
            try:
                obj.close()
            except Exception:
                pass
