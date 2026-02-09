from typing import List, Tuple
import numpy as np
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.reference_methods.predict_top_ms2deepscores import select_inchikeys_with_highest_ms2deepscore
from ms2query.benchmarking.TopKTanimotoScores import TopKTanimotoScores
from ms2query.metrics import generalized_tanimoto_similarity_matrix


def predict_using_closest_tanimoto(
    library_spectra: AnnotatedSpectrumSet,
    query_spectra: AnnotatedSpectrumSet,
    library_fingerprints: Fingerprints,
    nr_of_closest_inchikeys_to_select=10,
    nr_of_inchikeys_with_highest_ms2deepscore_to_select=100,
) -> Tuple[List[str], List[float]]:
    """Predict best inchikey, by taking the average score over all spectra for the 10 closest related library inchikeys.
    (simplified version of old MS2Query)
    """
    top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
        library_fingerprints,
        library_fingerprints,
        k=nr_of_closest_inchikeys_to_select,
    )
    ms2deepscores = cosine_similarity_matrix(query_spectra.embeddings.embeddings, library_spectra.embeddings.embeddings)
    inchikeys_with_highest_ms2deepscores = select_inchikeys_with_highest_ms2deepscore(
        query_spectra, library_spectra, nr_of_inchikeys_with_highest_ms2deepscore_to_select, ms2deepscores=ms2deepscores
    )

    inchikeys_of_best_match = []
    highest_scores = []
    for spectrum_idx in tqdm(range(len(query_spectra.spectra)), "Predicting using closest tanimoto"):
        average_predicted_scores = {}
        for inchikey in inchikeys_with_highest_ms2deepscores[spectrum_idx]:
            top_k_inchikeys = top_k_tanimoto_scores.select_top_k_inchikeys(inchikey)

            average_predicted_score = get_average_predictions_for_closely_related_metabolites(
                library_spectra, top_k_inchikeys, ms2deepscores[spectrum_idx]
            )
            average_predicted_scores[inchikey] = average_predicted_score

        inchikey_with_highest_average_prediction, score = max(
            average_predicted_scores.items(), key=lambda item: item[1]
        )
        inchikeys_of_best_match.append(inchikey_with_highest_average_prediction)
        highest_scores.append(score)
    return inchikeys_of_best_match, highest_scores


def get_average_predictions_for_closely_related_metabolites(
    spectra: AnnotatedSpectrumSet, top_k_inchikeys, all_ms2deepscores: np.ndarray
):
    """Calculates the average ms2deepscore predictions for top k closest inchikeys"""
    average_predicted_scores = []
    for top_inchikey in top_k_inchikeys:
        matching_spectrum_indexes = list(spectra.spectrum_indices_per_inchikey[top_inchikey])
        predicted_scores = all_ms2deepscores[matching_spectrum_indexes]
        average_predicted_scores.append(predicted_scores.mean())
    average_predicted_score = sum(average_predicted_scores) / len(average_predicted_scores)
    return average_predicted_score


def get_inchikey_and_tanimoto_scores_for_top_k(
    fingerprints: Fingerprints, inchikey: str, k: int
) -> tuple[list[str], np.ndarray]:
    """For an inchikey in a library the top k highest tanimoto scores in the library are predicted (including itself)"""
    similarity_scores = generalized_tanimoto_similarity_matrix(
        fingerprints.get_fingerprints([inchikey]), fingerprints.fingerprints
    )[0]
    inchikey_indexes_of_top_k = np.argpartition(similarity_scores, -k)[-k:]
    tanimoto_scores_for_top_k = similarity_scores[inchikey_indexes_of_top_k]

    top_inchikeys = [fingerprints.inchikeys[inchikey_index] for inchikey_index in inchikey_indexes_of_top_k]
    return top_inchikeys, tanimoto_scores_for_top_k
