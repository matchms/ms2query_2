import pytest
from ms2query.benchmarking.EvaluateMethods import EvaluateMethods
from ms2query.benchmarking.reference_methods.predict_best_possible_match import predict_best_possible_match
from ms2query.benchmarking.reference_methods.predict_highest_cosine import predict_highest_cosine
from ms2query.benchmarking.reference_methods.predict_highest_ms2deepscore import predict_highest_ms2deepscore
from ms2query.benchmarking.SpectrumDataSet import SpectraWithMS2DeepScoreEmbeddings
from tests.conftest import create_test_spectra, ms2deepscore_model


@pytest.mark.parametrize(
    "method",
    [predict_highest_ms2deepscore, predict_highest_cosine, predict_best_possible_match],
)
def test_evaluate_methods(method):
    nr_of_spectra_per_inchikey = 6
    nr_of_inchikeys = 5
    dummy_spectra = create_test_spectra(nr_of_spectra_per_inchikey, nr_of_inchikeys=nr_of_inchikeys)
    for i, spectrum in enumerate(dummy_spectra):
        if i % 3 == 0:
            spectrum.set("ionmode", "positive")
        else:
            spectrum.set("ionmode", "negative")
    model = ms2deepscore_model()
    reference_library = SpectraWithMS2DeepScoreEmbeddings(dummy_spectra[: nr_of_spectra_per_inchikey * 2], model)
    validation_spectra = SpectraWithMS2DeepScoreEmbeddings(dummy_spectra[nr_of_spectra_per_inchikey * 2 :], model)
    method_evaluator = EvaluateMethods(reference_library, validation_spectra)
    # should be zero or below zero, since it is the difference with the perfect predictions
    assert method_evaluator.benchmark_analogue_search(method) >= 0.0
    # # Should be 1 because we added a good match for each.
    assert method_evaluator.benchmark_exact_matching_within_ionmode(method, "positive") == 1.0
    assert method_evaluator.exact_matches_across_ionization_modes(method) == 1.0
