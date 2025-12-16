from typing import List, Tuple
from matchms import Scores
from matchms.similarity.CosineGreedy import CosineGreedy
from matchms.similarity.PrecursorMzMatch import PrecursorMzMatch
from ms2query.benchmarking.SpectrumDataSet import SpectraWithFingerprints


def predict_highest_cosine(
    library_spectra: SpectraWithFingerprints, query_spectra: SpectraWithFingerprints
) -> Tuple[List[str], List[float]]:

    scores = Scores(references=library_spectra.spectra, queries=query_spectra.spectra, is_symmetric=False)
    scores = scores.calculate(PrecursorMzMatch(0.1))
    scores = scores.calculate(CosineGreedy(tolerance=0.1))
    inchikeys_of_best_match = []
    highest_scores = []
    for query_spectrum in query_spectra.spectra:
        results = scores.scores_by_query(query_spectrum, "CosineGreedy_score", sort=True)
        if len(results) == 0:
            inchikeys_of_best_match.append(None)
            highest_scores.append(0.0)
        else:
            best_reference, highest_score = results[0]
            inchikeys_of_best_match.append(best_reference.get("inchikey")[:14])
            highest_scores.append(highest_score["CosineGreedy_score"])
    return inchikeys_of_best_match, highest_scores
