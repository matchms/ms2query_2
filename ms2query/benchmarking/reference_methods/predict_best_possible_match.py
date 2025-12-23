from typing import Dict
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet


def predict_best_possible_match(library_spectra: AnnotatedSpectrumSet, query_spectra: AnnotatedSpectrumSet):
    highest_possible_score_per_inchikey = calculate_highest_tanimoto_score_per_inchikey(library_spectra, query_spectra)

    inchikeys_of_best_match = []
    highest_scores = []

    for spectrum in query_spectra.spectra:
        inchikey = spectrum.get("inchikey")[:14]

        inchikeys_of_best_match.append(highest_possible_score_per_inchikey[inchikey][0])
        highest_scores.append(highest_possible_score_per_inchikey[inchikey][1])

    return inchikeys_of_best_match, highest_scores


def calculate_highest_tanimoto_score_per_inchikey(
    library_spectra: AnnotatedSpectrumSet, query_spectra: AnnotatedSpectrumSet
) -> Dict[str, tuple[str, float]]:
    """Finds the best possible match during an analogue search"""
    print("Calculating tanimoto scores to determine best possible match")

    library_fingerprints = library_spectra.fingerprints.fingerprints
    query_fingerprints = query_spectra.fingerprints.fingerprints
    tanimoto_scores = jaccard_similarity_matrix(library_fingerprints, query_fingerprints)
    highest_scores = tanimoto_scores.max(axis=0, initial=0)
    indexes_of_highest_scores = tanimoto_scores.argmax(axis=0)

    highest_possible_score_per_inchikey = dict()
    for i, inchikey in enumerate(query_spectra.fingerprints.inchikeys):
        # Check if inchikey in library (To correctly handle the exact matching case)
        if inchikey in library_spectra.fingerprints.inchikeys:
            highest_possible_score_per_inchikey[inchikey] = (inchikey, 1.0)
            continue

        highest_possible_score_per_inchikey[inchikey] = (
            library_spectra.fingerprints.inchikeys[indexes_of_highest_scores[i]],
            highest_scores[i],
        )
    return highest_possible_score_per_inchikey
