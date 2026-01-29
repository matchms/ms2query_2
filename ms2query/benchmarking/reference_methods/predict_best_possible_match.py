from typing import Dict
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Fingerprints import Fingerprints


def predict_best_possible_match(library_spectra: AnnotatedSpectrumSet,
                                query_spectra: AnnotatedSpectrumSet,
                                fingerprints: Fingerprints,):
    highest_possible_score_per_inchikey = calculate_highest_tanimoto_score_per_inchikey(library_spectra, query_spectra, fingerprints)

    inchikeys_of_best_match = []
    highest_scores = []

    for spectrum in query_spectra.spectra:
        inchikey = spectrum.get("inchikey")[:14]

        inchikeys_of_best_match.append(highest_possible_score_per_inchikey[inchikey][0])
        highest_scores.append(highest_possible_score_per_inchikey[inchikey][1])

    return inchikeys_of_best_match, highest_scores


def calculate_highest_tanimoto_score_per_inchikey(
    library_spectra: AnnotatedSpectrumSet, query_spectra: AnnotatedSpectrumSet,
        fingerprints: Fingerprints
) -> Dict[str, tuple[str, float]]:
    """Finds the best possible match during an analogue search"""
    print("Calculating tanimoto scores to determine best possible match")


    library_fingerprints = fingerprints.get_fingerprints(library_spectra.inchikeys)
    query_fingerprints = fingerprints.get_fingerprints(query_spectra.inchikeys)

    tanimoto_scores = jaccard_similarity_matrix(library_fingerprints, query_fingerprints)
    highest_scores = tanimoto_scores.max(axis=0, initial=0)
    indexes_of_highest_scores = tanimoto_scores.argmax(axis=0)

    highest_possible_score_per_inchikey = dict()
    # todo replace with TopKTanimotoScores
    inchikeys_in_library = set(library_spectra.inchikeys)
    for i, inchikey in enumerate(query_spectra.inchikeys):
        # Check if inchikey in library (To correctly handle the exact matching case)
        if inchikey in inchikeys_in_library:
            highest_possible_score_per_inchikey[inchikey] = (inchikey, 1.0)
            continue

        highest_possible_score_per_inchikey[inchikey] = (
            library_spectra.inchikeys[indexes_of_highest_scores[i]],
            highest_scores[i],
        )
    return highest_possible_score_per_inchikey
