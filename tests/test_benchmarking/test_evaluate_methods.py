from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.EvaluateAnalogueSearch import EvaluateAnalogueSearch
from tests.conftest import create_test_spectra, ms2deepscore_model


def create_dummy_library_and_validation_spectra() -> tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
    nr_of_spectra_per_inchikey = 6
    nr_of_inchikeys = 5
    dummy_spectra = create_test_spectra(nr_of_spectra_per_inchikey, nr_of_inchikeys=nr_of_inchikeys)
    for i, spectrum in enumerate(dummy_spectra):
        if i % 3 == 0:
            spectrum.set("ionmode", "positive")
        else:
            spectrum.set("ionmode", "negative")
    model = ms2deepscore_model()

    reference_library = AnnotatedSpectrumSet.create_spectrum_set(dummy_spectra[: nr_of_spectra_per_inchikey * 2])
    validation_spectra = AnnotatedSpectrumSet.create_spectrum_set(dummy_spectra[nr_of_spectra_per_inchikey * 2 :])
    reference_library.add_embeddings(model)
    validation_spectra.add_embeddings(model)
    return reference_library, validation_spectra


def test_evaluate_analogue_search():
    reference_library, validation_spectra = create_dummy_library_and_validation_spectra()
    method_evaluator = EvaluateAnalogueSearch(reference_library, validation_spectra)
    fake_predicted_inchikeys = [
        reference_library.inchikeys[i % len(reference_library.inchikeys)]
        for i in range(len(validation_spectra.spectra))
    ]
    method_evaluator.benchmark_analogue_search(fake_predicted_inchikeys)
