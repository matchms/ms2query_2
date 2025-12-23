from typing import List, Tuple
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.reference_methods.PredictMS2DeepScoreSimilarity import predict_top_ms2deepscores
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet


def predict_with_integrated_similarity_flow(
    library_spectra: AnnotatedSpectrumSet,
    query_spectra: AnnotatedSpectrumSet,
    number_of_analogues_to_consider=50,
) -> Tuple[List[str], List[float]]:

    all_indexes_of_library_spectra_with_highest_score, all_predicted_scores = predict_top_ms2deepscores(
        library_spectra.embeddings, query_spectra.embeddings, k=number_of_analogues_to_consider
    )
    inchikeys_of_best_matches = []
    highest_isf_scores = []
    # loop over the query spectra:
    for query_index in tqdm(range(len(query_spectra.spectra)), "Calculating ISF score"):
        highest_isf_score, inchikey_of_highest_isf_score = get_highest_isf(
            library_spectra,
            all_indexes_of_library_spectra_with_highest_score[query_index],
            all_predicted_scores[query_index],
        )
        inchikeys_of_best_matches.append(inchikey_of_highest_isf_score)
        highest_isf_scores.append(highest_isf_score)
    return inchikeys_of_best_matches, highest_isf_scores


def get_highest_isf(
    library_spectra: AnnotatedSpectrumSet,
    indexes_of_library_spectra_with_highest_score: np.ndarray,
    predicted_scores: [List[float]],
):

    # Get the corresponding inchikeys
    inchikeys_with_highest_ms2deepscore = [
        library_spectra.spectra[index].get("inchikey")[:14] for index in indexes_of_library_spectra_with_highest_score
    ]
    unique_inchikeys, average_scores, nr_of_spectra_per_inchikey = average_scores_per_inchikeys(
        predicted_scores, inchikeys_with_highest_ms2deepscore
    )
    # calculate tanimoto scores
    tanimoto_scores = jaccard_similarity_matrix(library_spectra.fingerprints.fingerprints, library_spectra.fingerprints.fingerprints)

    isf_scores = integrated_similarity_flow(average_scores, tanimoto_scores, nr_of_spectra_per_inchikey)
    index_of_highest_score = np.argmax(isf_scores)
    highest_isf_score = isf_scores[index_of_highest_score]
    inchikey_of_highest_isf_score = unique_inchikeys[index_of_highest_score]
    return highest_isf_score, inchikey_of_highest_isf_score


def average_scores_per_inchikeys(predicted_scores, inchikeys):
    """Calculate the average precicted score per inchikey
    This helps speed up the computations"""
    if len(predicted_scores) != len(inchikeys):
        raise ValueError
    scores_per_inchikey = {}
    for i, score in enumerate(predicted_scores):
        inchikey = inchikeys[i]
        if inchikey in scores_per_inchikey:
            scores_per_inchikey[inchikey].append(score)
        else:
            scores_per_inchikey[inchikey] = [score]
    # Take the average over the scores per inchikey
    unique_inchikeys = []
    average_scores = []
    nr_of_spectra_per_inchikey = []
    for inchikey in scores_per_inchikey:
        unique_inchikeys.append(inchikey)
        average_scores.append(sum(scores_per_inchikey[inchikey]) / len(scores_per_inchikey[inchikey]))
        nr_of_spectra_per_inchikey.append(len(scores_per_inchikey[inchikey]))
    return unique_inchikeys, average_scores, nr_of_spectra_per_inchikey


def integrated_similarity_flow(
    predicted_scores: List[float], similarities: np.ndarray, nr_of_spectra_per_inchikey: List[float]
) -> List[float]:
    """Compute the confidence of the prediction for each candidate.
    Integrated similarity flow (ISF) scores are calculated using the similarity of candidates among each other
    and their distance to the query spectrum.

    Args:
        distances (list): Distances of the candidates to the query spectrum in the chemical space.
        similarities (list of lists): Jaccard similarity of all candidates to each other.

    Returns:
        dict[int, float]: ISF scores for each candid+ate.
    """
    num_hits = len(predicted_scores)
    isf_scores = []

    # Total similarity
    total_similarity = sum([predicted_scores[i] * nr_of_spectra_per_inchikey[i] for i in range(len(predicted_scores))])

    for i in range(num_hits):
        isf_score = (
            sum(predicted_scores[j] * similarities[i][j] * nr_of_spectra_per_inchikey[j] for j in range(num_hits))
            / total_similarity
        )
        isf_scores.append(isf_score)

    return isf_scores
