import numpy as np
import pytest
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.reference_methods.predict_best_possible_match import predict_best_possible_match
from ms2query.benchmarking.reference_methods.predict_highest_ms2deepscore import predict_highest_ms2deepscore
from tests.helper_functions import (
    get_library_and_test_spectra_exactly_matching,
    get_library_and_test_spectra_not_identical,
)


@pytest.mark.parametrize(
    "prediction_function",
    [
        predict_highest_ms2deepscore,
    ],
)
def test_all_methods(prediction_function):
    library_spectra, test_spectra = get_library_and_test_spectra_exactly_matching()
    predicted_inchikeys, scores = prediction_function(library_spectra, test_spectra)
    for i, spectrum in enumerate(test_spectra.spectra):
        inchikey = spectrum.get("inchikey")[:14]  # type: ignore
        assert predicted_inchikeys[i] == inchikey
        assert np.allclose(scores[i], np.array(1.0), atol=1e-5)


def test_predict_best_possible_match():
    library_spectra, test_spectra = get_library_and_test_spectra_not_identical()
    fingerprints = Fingerprints.from_spectrum_set(library_spectra + test_spectra, "daylight", 2048)
    predicted_inchikeys, scores = predict_best_possible_match(library_spectra, test_spectra, fingerprints)
    for i, spectrum in enumerate(test_spectra.spectra):
        inchikey = spectrum.get("inchikey")[:14]  # type: ignore
        assert predicted_inchikeys[i] == inchikey
        assert np.allclose(scores[i], np.array(1.0), atol=1e-5)
