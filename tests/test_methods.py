import numpy as np
import pytest
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from ms2query.benchmarking.reference_methods.predict_best_possible_match import predict_best_possible_match
from ms2query.benchmarking.reference_methods.predict_highest_cosine import predict_highest_cosine
from ms2query.benchmarking.reference_methods.predict_highest_ms2deepscore import predict_highest_ms2deepscore
from ms2query.benchmarking.reference_methods.predict_with_integrated_similarity_flow import (
    integrated_similarity_flow,
    predict_with_integrated_similarity_flow,
)
from ms2query.benchmarking.AnnotatedSpectrumSet import SpectraWithMS2DeepScoreEmbeddings
from tests.conftest import create_test_spectra, ms2deepscore_model


@pytest.mark.parametrize(
    "prediction_function",
    [
        predict_highest_cosine,
        predict_highest_ms2deepscore,
        predict_best_possible_match,
    ],
)
def test_all_methods(prediction_function):
    model = ms2deepscore_model()
    library_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(), model)
    test_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(1), model)
    predicted_inchikeys, scores = prediction_function(library_spectra, test_spectra)
    for i, spectrum in enumerate(test_spectra.spectra):
        inchikey = spectrum.get("inchikey")[:14]
        assert predicted_inchikeys[i] == inchikey
        assert np.allclose(scores[i], np.array(1.0), atol=1e-5)


def test_predict_with_integrated_similarity_flow():
    model = ms2deepscore_model()
    library_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(), model)
    test_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(1), model)
    predicted_inchikeys, scores = predict_with_integrated_similarity_flow(library_spectra, test_spectra)

    assert predicted_inchikeys == ["RYYVLZVUVIJVGH", "ZPUCINDJVBIVPJ", "ZPUCINDJVBIVPJ"]
    assert np.allclose(np.array([0.38829751082577607, 0.3919729335980483, 0.38774130710967564]), np.array(scores))


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
