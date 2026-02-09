from typing import List, Tuple
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.reference_methods.predict_top_ms2deepscores import predict_top_ms2deepscores


def predict_highest_ms2deepscore(
    library_spectra: AnnotatedSpectrumSet, query_spectra: AnnotatedSpectrumSet
) -> Tuple[List[str], List[float]]:
    indexes_of_highest_scores, highest_scores = predict_top_ms2deepscores(
        library_spectra.embeddings, query_spectra.embeddings, k=1
    )
    single_highest_score = [highest_score[0] for highest_score in highest_scores]
    inchikeys_of_best_match = [
        library_spectra.spectra[index[0]].get("inchikey")[:14] for index in indexes_of_highest_scores
    ]
    return inchikeys_of_best_match, single_highest_score
