from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from matchms import Spectrum
from ms2deepscore.models import load_model as _ms2ds_load_model
from ms2query import MS2QueryDatabase
from ms2query.data_processing import compute_spectra_embeddings
from ms2query.database import EmbeddingIndex, FingerprintSparseIndex


@dataclass
class MS2QueryLibrary:
    """
    Central glue/hub for MS2Query actions.

    Owns:
      - MS2QueryDatabase (SQLite-backed spectra/compounds/map)
      - EmbeddingIndex (dense cosine ANN over MS2DeepScore embeddings, returns spec_id)
      - optional FingerprintSparseIndex (for "all_compounds" search by fingerprints)

    Also provides:
      - process_spectra(spectra) -> spectra   (placeholder; hook for future matchms filters)
      - compute_embeddings(spectra) -> np.ndarray[float32]  (L2-normalized)
      - query_embedding_index(spectra, k=10) -> List[List[Dict]] with {'spec_id','score','rank'}

    Notes
    -----
    * `model_path` is optional; provide it if you want on-the-fly embedding of ad-hoc spectra.
      If omitted, you can still query using precomputed embeddings fetched from SQLite.
    * EmbeddingIndex must be built/loaded elsewhere (creation handled by setup workflow).
    """
    db: MS2QueryDatabase
    embedding_index: Optional[EmbeddingIndex] = None
    fingerprint_index: Optional[FingerprintSparseIndex] = None  # for now: reference spectra only
    model_path: Optional[str] = None

    # internal: whether to apply spectrum normalization (sum=1) before embedding
    _spectrum_sum_normalization_for_embedding: bool = True

    # cached MS2DeepScore model
    _model: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle / internal helpers
    # -----------------------------------------------------------------

    def _ensure_model(self):
        """Lazy-load MS2DeepScore model if a model_path was provided."""
        if self._model is None:
            if not self.model_path:
                raise RuntimeError(
                    "MS2QueryLibrary: model_path is not set; cannot compute embeddings on-the-fly."
                )
            # Using direct loader to avoid circular coupling here.
            self._model = _ms2ds_load_model(self.model_path)
            self._model.eval()
        return self._model
    
    def _ensure_embedding_index(self):
        if self.embedding_index is None:
            raise RuntimeError(
                "EmbeddingIndex is not set. Build or load it before querying."
            )

    def _ensure_fingerprint_index(self):
        if self.fingerprint_index is None:
            raise RuntimeError(
                "FingerprintSparseIndex is not set. Build or load it before querying."
            )

    # ------------------------------------------------------------------
    # Core API: spectra -> embeddings -> ANN over embeddings
    # -----------------------------------------------------------------

    def process_spectra(self, spectra: list[Spectrum]) -> List[Spectrum]:
        """
        Placeholder for your future matchms filter pipeline.
        For now: return spectra as-is (but ensure ms2query normalization during embedding).
        """
        # Hook point: insert matchms pipeline later (e.g., metadata fixes, peak processing, etc.)
        return list(spectra)

    def compute_embeddings(self, spectra: Sequence[Spectrum]) -> np.ndarray:
        """
        Compute MS2DeepScore embeddings for arbitrary query spectra.

        Spectra will be preprocessed via self.process_spectra(...) first.
        """
        spectra = _ensure_spectra_list(spectra)
        if not spectra:
            return np.empty((0, 0), dtype=np.float32)

        model = self._ensure_model()
        spectra = self.process_spectra(spectra)

        return compute_spectra_embeddings(
            model,
            spectra,
            normalize_spectrum=self._spectrum_sum_normalization_for_embedding,
        )

    def query_embedding_index(
        self,
        spectra: Union[Spectrum, Sequence[Spectrum]],
        *,
        k: int = 10,
        ef: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> Union[List[List[Dict[str, Any]]], "pd.DataFrame"]:
        """
        Process spectra -> embed -> query EmbeddingIndex.

        Returns per-query results as a list of lists with dicts:
            [{'rank': 1, 'spec_id': '...', 'score': float}, ...]
        All IDs are **spec_id** strings (NOT internal index ids).

        Parameters
        ----------
        spectra : Spectrum | Sequence[Spectrum]
            Query spectra.
        k : int
            Top-k to return.
        ef : Optional[int]
            nmslib ef (higher = better recall / slower).
        return_dataframe : bool
            If True, returns a tidy DataFrame with columns:
              ['query_ix', 'rank', 'spec_id', 'score']
        """
        self._ensure_embedding_index()
        spectra = _ensure_spectra_list(spectra)
        embeddings = self.compute_embeddings(spectra)

        if embeddings.size == 0:
            return (
                [] if not return_dataframe else self._empty_result_df()
            )

        # Batched call: EmbeddingIndex.query returns List[List[(spec_id, score)]]
        batch_hits = self.embedding_index.query(embeddings, k=k, ef=ef)

        results_all: List[List[Dict[str, Any]]] = []
        for hits in batch_hits:
            one = [
                {"rank": rk + 1, "spec_id": spec_id, "score": float(score)}
                for rk, (spec_id, score) in enumerate(hits)
            ]
            results_all.append(one)

        if not return_dataframe:
            return results_all

        rows: List[Dict[str, Any]] = []
        for qi, lst in enumerate(results_all):
            for item in lst:
                rows.append({"query_ix": qi, **item})
        return pd.DataFrame(rows, columns=["query_ix", "rank", "spec_id", "score"])

    def query_spectra_by_spectra(
        self,
        spectra: Union[Spectrum, Sequence[Spectrum]],
        *,
        k_spectra: int = 10,
        ef: Optional[int] = None,
    ):
        """
        Query the embedding index with spectra, return top-k_spectra per spectrum.

        Parameters
        ----------
        spectra : list[Spectrum] or Spectrum
            Query spectra.
        k_spectra : int
            Number of top spectra to retrieve from the embedding index.
        ef : Optional[int]
            nmslib ef parameter (higher = better recall / slower).
        """
        return self.query_embedding_index(
            spectra, k=k_spectra, ef=ef, return_dataframe=True
        )

    # ------------------------------------------------------------------
    # Core API: compounds / fingerprints
    # ------------------------------------------------------------------

    def query_compounds_by_compounds(
        self,
        compounds: Sequence[str],
        *,
        k_compounds: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """
        Query the fingerprint index with compounds, return top-k compounds per compound.

        Parameters
        ----------
        compounds : Sequence[str]
            Query compounds (expects list of SMILES strings).
        k_compounds : int
            Number of top compounds to return per query compound.
        """
        self._ensure_fingerprint_index()

        # Compute fingerprints (sparse representation)
        fps = self.db.all_cdb.compute_fingerprints(
            compounds,
            count=False,
            sparse=True,
        )

        # Batched fingerprint ANN query
        batch_hits = self.fingerprint_index.query(fps, k=k_compounds)

        results_all: List[List[Dict[str, Any]]] = []
        for hits in batch_hits:
            one = [
                {"rank": rk + 1, "comp_id": comp_id, "score": float(score)}
                for rk, (comp_id, score) in enumerate(hits)
            ]
            results_all.append(one)

        return results_all
 
    def query_compounds_by_spectra(
        self,
        spectra: Union[Spectrum, Sequence[Spectrum]],
        *,
        k_spectra: int = 100,
        k_compounds: int = 10,
        ef: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query the embedding index with spectra, then aggregate to compounds.

        Parameters
        ----------
        spectra : list[Spectrum] or Spectrum
            Query spectra.
        k_spectra : int
            Number of top spectra to retrieve from the embedding index.
        k_compounds : int
            Number of top compounds to return per query spectrum.
        ef : Optional[int]
            nmslib ef parameter (higher = better recall / slower).
        """
        if k_compounds > k_spectra:
            raise ValueError("k_compounds cannot be larger than k_spectra")

        # Query spectral embeddings
        results = self.query_spectra_by_spectra(
            spectra, k_spectra=k_spectra, ef=ef
        )  # DataFrame

        if results.empty:
            return results

        # Pick k_compounds top compounds from the k_spectra hits (if possible)
        spec_ids = results["spec_id"].values

        compounds = (
            self.db.metadata_by_spec_ids(list(spec_ids))
            .set_index("spec_id")
        )

        compounds = (
            compounds.merge(results, on="spec_id")
            .sort_values(["query_ix", "rank"])
        )

        # Pick no more than k_compounds per query_ix
        idx = compounds.groupby(["query_ix", "rank"])["score"].idxmax()
        best_per_pair = compounds.loc[idx]

        # Within each query_ix, keep the top-k by score
        df_selected = (
            best_per_pair.sort_values(["query_ix", "score"], ascending=[True, False])
            .groupby("query_ix", group_keys=False)
            .head(k_compounds)
            .reset_index(drop=True)
        )
        return df_selected

    def analogue_search(
        self,
        spectra: Union[Spectrum, Sequence[Spectrum]],
        *,
        k_spectra: int = 1,
        k_compounds: int = 10,
        ef: Optional[int] = None,
    ):
        """
        Perform an analogue search for the given spectra.

        Current behaviour:
        - For each query spectrum, retrieve top-`k_spectra` library spectra.
        - Get their compounds.
        - Run compound-by-compound search in fingerprint space.
        """
        # Step 1: top-k_spectra per query
        spec_hits = self.query_spectra_by_spectra(
            spectra, k_spectra=k_spectra, ef=ef
        )  # DataFrame
        if spec_hits.empty:
            return []

        spec_ids = spec_hits["spec_id"].values

        # Step 2: get compounds of all retrieved spectra
        analogue_compounds = (
            self.db.metadata_by_spec_ids(list(spec_ids))
            .set_index("spec_id")
        )

        smiles = analogue_compounds["smiles"].tolist()

        # Step 3: fingerprint-based compound search
        top_compounds = self.query_compounds_by_compounds(
            smiles, k_compounds=k_compounds
        )
        return top_compounds

    # ------------------------------------------------------------------
    # Helpers / glue
    # ------------------------------------------------------------------

    def set_embedding_index(self, index: EmbeddingIndex) -> None:
        """Attach or replace the EmbeddingIndex."""
        self.embedding_index = index

    def set_fingerprint_index(self, index: FingerprintSparseIndex) -> None:
        """Attach or replace the sparse fingerprint index (optional)."""
        self.fingerprint_index = index

    def query_by_spec_ids(
        self,
        spec_ids: List[str],
        *,
        k: int = 10,
        ef: Optional[int] = None,
        return_dataframe: bool = False,
    ):
        """
        Convenience: fetch embeddings for known spec_ids from SQLite and search.
        Requires that the embeddings are present in DB (table 'embeddings').
        """
        self._ensure_embedding_index()

        # Pull precomputed embeddings from DB (already L2-normalized)
        ids, X = self.db.ref_sdb.get_embeddings(
            ids=spec_ids,
            embeddings_table="embeddings",
            normalized=True,
        )

        # If DB returns nothing, keep the old API behaviour
        if X.size == 0:
            return [] if not return_dataframe else self._empty_result_df()

        # X is 2D: use batched query
        batch_hits = self.embedding_index.query(X, k=k, ef=ef)

        results_all: List[List[Dict[str, Any]]] = []
        for hits in batch_hits:
            one = [
                {"rank": rk + 1, "spec_id": sid, "score": float(score)}
                for rk, (sid, score) in enumerate(hits)
            ]
            results_all.append(one)

        if not return_dataframe:
            return results_all

        rows: List[Dict[str, Any]] = []
        for qi, lst in enumerate(results_all):
            for item in lst:
                rows.append({"query_ix": qi, **item})
        return pd.DataFrame(rows, columns=["query_ix", "rank", "spec_id", "score"])

    @staticmethod
    def _empty_result_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["query_ix", "rank", "spec_id", "score"])


# ----------------- helper functions ---------------------

def _ensure_spectra_list(
    spectra: Union[Spectrum, Sequence[Spectrum]]
) -> List[Spectrum]:
    if isinstance(spectra, Spectrum):
        return [spectra]
    if isinstance(spectra, Sequence):
        return list(spectra)
    raise ValueError("spectra must be a Spectrum or a sequence of Spectrum objects.")
