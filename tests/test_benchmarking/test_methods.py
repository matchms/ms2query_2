import numpy as np
import pytest
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.reference_methods.predict_best_possible_match import predict_best_possible_match
from ms2query.benchmarking.reference_methods.predict_highest_ms2deepscore import predict_highest_ms2deepscore
from ms2query.benchmarking.reference_methods.predict_with_integrated_similarity_flow import (
    integrated_similarity_flow,
    predict_with_integrated_similarity_flow,
)
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


def test_predict_with_integrated_similarity_flow():
    library_spectra, test_spectra = get_library_and_test_spectra_not_identical()
    fingerprints = Fingerprints.from_spectrum_set(library_spectra, "daylight", 4096)
    predicted_inchikeys, scores = predict_with_integrated_similarity_flow(library_spectra, test_spectra, fingerprints)


def test_isf_computation():
    distances = [0.99, 0.99, 0.99, 0.5, 0.5]
    fps = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0]])
    similarities = jaccard_similarity_matrix(fps, fps)
    result = integrated_similarity_flow(distances, similarities, [1, 1, 1, 1, 1])
    expected_result = [
        0.715869,
        0.686481,
        0.736523,
        0.492107,
        0.359110,
    ]
    assert np.allclose(np.array(expected_result), np.array(result))
