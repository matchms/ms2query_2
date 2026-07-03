import numpy as np
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.Fingerprints import Fingerprints
from ms2query.ms2query_development.reference_methods.predict_top_k_ms2deepscore import (
    select_inchikeys_with_highest_ms2deepscore,
)
from ms2query.ms2query_development.TopKTanimotoScores import TopKTanimotoScores


def calculate_MS2DeepScoresForTopKInChikeys_from_spectra(
    library_spectra: AnnotatedSpectrumSet,
    query_spectra: AnnotatedSpectrumSet,
    fingerprint_type: str,
    fingerprint_nbits,
    nr_of_closest_inchikeys_to_select=10,
    nr_of_inchikeys_with_highest_ms2deepscore_to_select=100,
) -> list[dict[str, "MS2DeepScoresForTopKInChikeys"]]:
    """A wrapper to do all the preprocessing steps to get the TopKMS2DeepScores for each query spectrum"""

    library_fingerprints = Fingerprints.from_spectrum_set(library_spectra, fingerprint_type, fingerprint_nbits)

    top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
        library_fingerprints,
        library_fingerprints,
        k=nr_of_closest_inchikeys_to_select,
    )

    inchikeys_with_highest_ms2deepscores = select_inchikeys_with_highest_ms2deepscore(
        query_spectra, library_spectra, nr_of_inchikeys_with_highest_ms2deepscore_to_select
    )

    return calculate_MS2DeepScoresForTopKInChikeys(
        library_spectra, query_spectra, top_k_tanimoto_scores, inchikeys_with_highest_ms2deepscores
    )


def calculate_MS2DeepScoresForTopKInChikeys(
    library_spectra: AnnotatedSpectrumSet,
    query_spectra: AnnotatedSpectrumSet,
    top_k_tanimoto_scores: TopKTanimotoScores,
    inchikeys_with_highest_ms2deepscores_per_query_spectrum: list[list[str]],
) -> list[dict[str, "MS2DeepScoresForTopKInChikeys"]]:
    """Gets all MS2DeepScores for the library inchikeys with a high score and their most similar inchikeys

    For each query spectrum the library inchikeys with the highest predicted MS2DeepScore are selected.
    For each of these library spectra, the closest library spectra are selected.

    inchikeys_with_highest_ms2deepscores_per_query_spectrum:
        The library inchikeys that have the highest ms2deepscore prediction for each query spectrum.
    top_k_tanimoto_scores:
        For each library inchikey the top k closest other library inchikeys on tanimoto scores.
    """
    assert len(inchikeys_with_highest_ms2deepscores_per_query_spectrum) == len(query_spectra)

    close_tanimoto_scores_per_inchikey_per_query_spectrum = []
    for spectrum_idx in tqdm(
        range(len(query_spectra.spectra)), "Calculating", total=len(query_spectra.spectra), ncols=200
    ):
        inchikeys_with_highest_ms2deepscores = inchikeys_with_highest_ms2deepscores_per_query_spectrum[spectrum_idx]
        close_tanimoto_scores_per_inchikey = {}
        for inchikey_with_high_ms2deepscore in inchikeys_with_highest_ms2deepscores:
            query_embedding = query_spectra.embeddings._embeddings[[spectrum_idx]]
            top_k_inchikeys_and_scores = top_k_tanimoto_scores.select_top_k_inchikeys_and_scores(
                inchikey_with_high_ms2deepscore
            )
            close_tanimoto_scores = MS2DeepScoresForTopKInChikeys(
                query_embedding, library_spectra, top_k_inchikeys_and_scores
            )

            close_tanimoto_scores_per_inchikey[inchikey_with_high_ms2deepscore] = close_tanimoto_scores
        close_tanimoto_scores_per_inchikey_per_query_spectrum.append(close_tanimoto_scores_per_inchikey)
    return close_tanimoto_scores_per_inchikey_per_query_spectrum


class MS2DeepScoresForTopKInChikeys:
    """Stores the MS2DeepScores and Tanimoto scores for the top k closest lib spectra

    This is only needed for the benchmarking and development (in the notebooks)
    and is not used for running the final verison of MS2Query

    This allows for quick testing of different reranking strategies. E.g. get_mean is similar to the original MS2Query,
    but it can also be used to make matrixes with both MS2DeepScore and tanimoto scores to train small reranking models.

    query_embedding: A single MS2DeepScore embedding stored as a 2D numpy array
    library_spectra: The reference spectra.
    top_k_inchikeys_and_scores: A dictionary with the closest inchikeys and corresponding tanimoto scores."""

    def __init__(
        self,
        query_embedding: np.ndarray,
        library_spectra: AnnotatedSpectrumSet,
        top_k_inchikeys_and_scores: dict[str, float],
    ):
        if len(query_embedding.shape) != 2 or query_embedding.shape[0] != 1:
            raise ValueError("Expected a single embedding, but as a 2D matrix. Like [[0, 1, 0,1],]")
        self._top_k_inchikeys_and_tanimoto_scores = top_k_inchikeys_and_scores

        self.ms2deepscores_per_inchikey = {}
        for top_inchikey in self._top_k_inchikeys_and_tanimoto_scores.keys():
            matching_spectrum_indexes = list(library_spectra.spectrum_indices_per_inchikey[top_inchikey])
            selected_lib_embeddings = library_spectra.embeddings._embeddings[matching_spectrum_indexes]
            ms2deepscores = cosine_similarity_matrix(query_embedding, selected_lib_embeddings)
            self.ms2deepscores_per_inchikey[top_inchikey] = ms2deepscores

    def get_mean_per_inchikey(self) -> dict[str, float]:
        average_predicted_ms2deepscores = {}
        for inchikey, predicted_ms2deepscores in self.ms2deepscores_per_inchikey.items():
            average_predicted_ms2deepscores[inchikey] = predicted_ms2deepscores.mean()
        return average_predicted_ms2deepscores

    def get_mean(self):
        """Returns the mean, over the mean ms2deepscore per inchikey"""
        mean_ms2deepscore_per_inchikey = self.get_mean_per_inchikey().values()
        return sum(mean_ms2deepscore_per_inchikey) / len(mean_ms2deepscore_per_inchikey)

    def get_max_per_inchikey(self) -> dict[str, float]:
        average_predicted_ms2deepscores = {}
        for inchikey, predicted_ms2deepscores in self.ms2deepscores_per_inchikey.items():
            average_predicted_ms2deepscores[inchikey] = predicted_ms2deepscores.max()
        return average_predicted_ms2deepscores

    @property
    def top_k_inchikeys_and_tanimoto_scores(self):
        return self._top_k_inchikeys_and_tanimoto_scores
