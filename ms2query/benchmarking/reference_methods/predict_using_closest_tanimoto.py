from typing import List, Tuple
import numpy as np
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.SpectrumDataSet import SpectrumSet
from ms2query.metrics import generalized_tanimoto_similarity_matrix


def predict_using_closest_tanimoto(
    library_spectra: SpectrumSet, query_spectra: SpectrumSet,
        nr_of_closest_inchikeys_to_select=10,
        nr_of_inchikeys_with_highest_ms2deepscore_to_select=100
) -> Tuple[List[str], List[float]]:
    """Predict best inchikey, by taking the average score over all spectra for the 10 closest related library inchikeys.
    (simplified version of old MS2Query)
    """
    inchikeys_of_best_match = []
    highest_scores = []
    for spectrum_idx in tqdm(range(len(query_spectra.spectra)), "Predicting using closest tanimoto"):
        inchikey_of_best_match, score = predict_using_closest_tanimoto_single_spectrum(
            library_spectra, query_spectra.subset_spectra([spectrum_idx]),
            nr_of_closest_inchikeys_to_select, nr_of_inchikeys_with_highest_ms2deepscore_to_select)
        inchikeys_of_best_match.append(inchikey_of_best_match)
        highest_scores.append(score)
    return inchikeys_of_best_match, highest_scores


def predict_using_closest_tanimoto_single_spectrum(
        spectra_with_embeddings: SpectrumSet, single_spectrum_with_embeddings: SpectrumSet,
        nr_of_closest_inchikeys_to_select, nr_of_inchikeys_with_highest_ms2deepscore_to_select) -> Tuple[str, float]:
    if len(single_spectrum_with_embeddings.spectra) != 1:
        raise ValueError("expected a single spectrum")
    ms2deepscores = cosine_similarity_matrix(single_spectrum_with_embeddings.embeddings.embeddings,
                                             spectra_with_embeddings.embeddings.embeddings)[0]
    top_inchikeys = select_inchikeys_with_highest_ms2deepscore(spectra_with_embeddings, ms2deepscores,
                                                               nr_of_inchikeys_with_highest_ms2deepscore_to_select)
    average_predicted_scores = {}
    for inchikey in top_inchikeys:
        top_k_inchikeys, _ = get_inchikey_and_tanimoto_scores_for_top_k(
            spectra_with_embeddings, inchikey, nr_of_closest_inchikeys_to_select)
        average_predicted_score = get_average_predictions_for_closely_related_metabolites(
            spectra_with_embeddings, top_k_inchikeys, ms2deepscores)
        average_predicted_scores[inchikey] = average_predicted_score

    inchikey_with_highest_average_prediction, score = max(average_predicted_scores.items(), key=lambda item: item[1])
    return inchikey_with_highest_average_prediction, score

def select_inchikeys_with_highest_ms2deepscore(spectra_with_embeddings: SpectrumSet, ms2deepscores, nr_of_inchikeys_to_select=10):
    highest_score_for_inchikey = []
    for inchikey, spectrum_indexes in spectra_with_embeddings.spectrum_indexes_per_inchikey.items():
        all_ms2deepscores_for_inchikey = ms2deepscores[spectrum_indexes]
        highest_score_for_inchikey.append(max(all_ms2deepscores_for_inchikey))
    inchikey_indexes_with_highest_ms2deepscore = np.argpartition(
        np.array(highest_score_for_inchikey), -nr_of_inchikeys_to_select)[-nr_of_inchikeys_to_select:]

    all_inchikeys = list(spectra_with_embeddings.most_common_inchi_per_inchikey.keys())
    top_inchikeys = [all_inchikeys[inchikey_index] for inchikey_index in inchikey_indexes_with_highest_ms2deepscore]
    return top_inchikeys

def get_average_predictions_for_closely_related_metabolites(spectra_with_embeddings, top_k_inchikeys,
                                                            all_ms2deepscores):
    """Calculates the average ms2deepscore predictions for top k closest inchikeys"""
    average_predicted_scores = []
    for top_inchikey in top_k_inchikeys:
        matching_spectrum_indexes = spectra_with_embeddings.spectrum_indexes_per_inchikey[top_inchikey]
        predicted_scores = all_ms2deepscores[matching_spectrum_indexes]
        average_predicted_scores.append(predicted_scores.mean())
    average_predicted_score = sum(average_predicted_scores) / len(average_predicted_scores)
    return average_predicted_score

def get_inchikey_and_tanimoto_scores_for_top_k(spectra: SpectrumSet, inchikey, k
                                               ) -> tuple[list[str], np.ndarray]:
    """For an inchikey in a library the top k highest tanimoto scores in the library are predicted (including itself)"""
    similarity_scores = generalized_tanimoto_similarity_matrix(spectra.fingerprints.get_fingerprints(inchikey), spectra.fingerprints.fingerprints)[0]
    inchikey_indexes_of_top_k = np.argpartition(similarity_scores, -k)[-k:]
    tanimoto_scores_for_top_k = similarity_scores[inchikey_indexes_of_top_k]

    top_inchikeys = [spectra.fingerprints.inchikeys[inchikey_index] for inchikey_index in inchikey_indexes_of_top_k]
    return top_inchikeys, tanimoto_scores_for_top_k
