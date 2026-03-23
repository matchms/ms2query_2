from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.MS2DeepScoresForTopInChikeys import (
    calculate_MS2DeepScoresForTopKInChikeys_from_spectra,
)
from tests.helper_functions import create_test_spectra, ms2deepscore_model


def test_calculate_MS2DeepScoresForTopKInChikeys_from_spectra():
    model = ms2deepscore_model()
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(create_test_spectra(nr_of_inchikeys=7))
    test_spectra = AnnotatedSpectrumSet.create_spectrum_set(create_test_spectra(1, nr_of_inchikeys=3))
    library_spectra.add_embeddings(model)
    test_spectra.add_embeddings(model)
    calculate_MS2DeepScoresForTopKInChikeys_from_spectra(library_spectra, test_spectra, "daylight", 100, 3, 3)
