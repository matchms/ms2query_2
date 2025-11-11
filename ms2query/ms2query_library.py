from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from matchms import Spectrum
from ms2deepscore.models import compute_embedding_array
from ms2deepscore.models import load_model as _ms2ds_load_model
from ms2query import MS2QueryDatabase
from ms2query.data_processing import normalize_spectrum_sum
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
    fingerprint_index: Optional[FingerprintSparseIndex] = None
    model_path: Optional[str] = None

    # internal: cached MS2DeepScore model
    _model: Any = field(default=None, init=False, repr=False)

    # ----------------------------- lifecycle -----------------------------

    def _ensure_model(self):
        """Lazy-load MS2DeepScore model if a model_path was provided."""
        if self._model is None:
            if not self.model_path:
                raise RuntimeError(
                    "MS2QueryLibrary: model_path is not set; cannot compute embeddings on-the-fly."
                )
            # You can load via SpectralDatabase helper or directly; both end up identical.
            # Using direct loader to avoid circular coupling here.
            self._model = _ms2ds_load_model(self.model_path)
            self._model.eval()
        return self._model

    # ----------------------------- core API -----------------------------

    def process_spectra(self, spectra: list[Spectrum]) -> List[Spectrum]:
        """
        Placeholder for your future matchms filter pipeline.
        For now: return spectra as-is (but ensure ms2query normalization during embedding).
        """
        # Hook point: insert matchms pipeline later (e.g., metadata fixes, peak processing, etc.)
        return list(spectra)

    def compute_embeddings(self, spectra: list[Spectrum], *, normalize_inputs: bool = True) -> np.ndarray:
        """
        Compute MS2DeepScore embeddings for arbitrary query spectra.

        - Applies ms2query's normalize_spectrum_sum() if normalize_inputs=True
        - Returns L2-normalized embeddings (float32) suitable for cosine ANN
        """
        if not spectra:
            return np.empty((0, 0), dtype=np.float32)

        model = self._ensure_model()

        # preprocess — keep spectral normalization symmetrical with DB embeddings
        proc = self.process_spectra(spectra)
        if normalize_inputs:
            proc = [normalize_spectrum_sum(s) for s in proc]

        E = compute_embedding_array(model, proc).astype(np.float32, copy=False)

        # L2 normalize (EmbeddingIndex assumes/benefits from cosine-normalized vectors)
        n = np.linalg.norm(E, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        E = E / n
        return E

    def query_embedding_index(
        self,
        spectra: Union[Spectrum, Sequence[Spectrum]],
        *,
        k: int = 10,
        ef: Optional[int] = None,
        assume_normalized: bool = True,
        return_dataframe: bool = True,
    ) -> Union[List[List[Dict[str, Any]]], "pd.DataFrame"]:
        """
        Process spectra -> embed -> query EmbeddingIndex.

        Returns per-query results as a list of lists with dicts:
            [{'rank':1, 'spec_id': '...', 'score': float}, ...]
        All IDs are **spec_id** strings (NOT internal index ids).

        Parameters
        ----------
        spectra : Spectrum | list[Spectrum]
            Query spectra.
        k : int
            Top-k to return.
        ef : Optional[int]
            nmslib ef (higher = better recall / slower).
        assume_normalized : bool
            If False, will L2-normalize vectors again before query (normally keep True).
        return_dataframe : bool
            If True, returns a tidy DataFrame with columns:
              ['query_ix','rank','spec_id','score']
        """
        if self.embedding_index is None:
            raise RuntimeError("EmbeddingIndex is not set. Build or load it before querying.")

        # Single → list
        if isinstance(spectra, Spectrum):
            spectra = [spectra]

        # Compute embeddings (L2-normalized)
        E = self.compute_embeddings(spectra)

        results_all: List[List[Dict[str, Any]]] = []
        for qi in range(E.shape[0]):
            # EmbeddingIndex.query returns list[(spec_id, similarity)]
            hits = self.embedding_index.query(E[qi], k=k, ef=ef, assume_normalized=assume_normalized)
            # convert to standard structure
            one = []
            for rk, (spec_id, score) in enumerate(hits, start=1):
                one.append({"rank": rk, "spec_id": spec_id, "score": float(score)})
            results_all.append(one)

        if not return_dataframe:
            return results_all

        rows = []
        for qi, lst in enumerate(results_all):
            for item in lst:
                rows.append({"query_ix": qi, **item})
        df = pd.DataFrame(rows, columns=["query_ix", "rank", "spec_id", "score"])
        return df

    # ----------------------------- helpers / optional glue -----------------------------

    def set_embedding_index(self, index: EmbeddingIndex) -> None:
        """Attach or replace the EmbeddingIndex."""
        self.embedding_index = index

    def set_fingerprint_index(self, index: FingerprintSparseIndex) -> None:
        """Attach or replace the sparse fingerprint index (optional)."""
        self.fingerprint_index = index

    def query_by_spec_ids(
        self, spec_ids: List[str], *, k: int = 10, ef: Optional[int] = None, return_dataframe: bool = False
    ):
        """
        Convenience: fetch embeddings for known spec_ids from SQLite and search.
        Requires that the embeddings are present in DB (table 'embeddings').
        """
        if self.embedding_index is None:
            raise RuntimeError("EmbeddingIndex is not set. Build or load it before querying.")

        # Pull precomputed embeddings from DB (already L2-normalized in SpectralDatabase.get_embeddings)
        ids, X = self.db.ref_sdb.get_embeddings(ids=spec_ids, embeddings_table="embeddings", normalized=True)
        if X.size == 0:
            return [] if not return_dataframe else self._empty_result_df()

        results_all: List[List[Dict[str, Any]]] = []
        for qi in range(X.shape[0]):
            hits = self.embedding_index.query(X[qi], k=k, ef=ef, assume_normalized=True)
            one = [{"rank": rk + 1, "spec_id": sid, "score": float(score)} for rk, (sid, score) in enumerate(hits)]
            results_all.append(one)

        if not return_dataframe:
            return results_all

        import pandas as pd
        rows = []
        for qi, lst in enumerate(results_all):
            for item in lst:
                rows.append({"query_ix": qi, **item})
        return pd.DataFrame(rows, columns=["query_ix", "rank", "spec_id", "score"])

    @staticmethod
    def _empty_result_df():
        import pandas as pd
        return pd.DataFrame(columns=["query_ix", "rank", "spec_id", "score"])
