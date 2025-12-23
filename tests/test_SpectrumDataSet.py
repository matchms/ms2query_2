from ms2query.benchmarking.AnnotatedSpectrumSet import (
    AnnotatedSpectrumSet,
)
from tests.conftest import create_test_spectra, ms2deepscore_model


def test_create_annotated_spectrum_set():
    test_spectra = create_test_spectra(nr_of_inchikeys=3, number_of_spectra_per_inchikey=3)
    spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(spectra=test_spectra)
    assert len(spectrum_set.spectrum_indexes_per_inchikey) == 3

def test_add_spectrum_sets():
    test_spectra = create_test_spectra(nr_of_inchikeys=3, number_of_spectra_per_inchikey=3)

    correct_combined_set = AnnotatedSpectrumSet.create_spectrum_set(test_spectra)

    spectrum_set_1 = AnnotatedSpectrumSet.create_spectrum_set(test_spectra[:5])
    spectrum_set_2 = AnnotatedSpectrumSet.create_spectrum_set(test_spectra[5:])

    # with added embededings
    model = ms2deepscore_model()
    spectrum_set_1.add_embeddings(model)
    spectrum_set_2.add_embeddings(model)
    correct_combined_set.add_embeddings(model)

    combined_spectra = spectrum_set_1 + spectrum_set_2
    assert correct_combined_set == combined_spectra

def test_subsetting():
    test_spectra = create_test_spectra(nr_of_inchikeys=3, number_of_spectra_per_inchikey=3)

    correct_subsetted_set = AnnotatedSpectrumSet.create_spectrum_set(test_spectra[:5])
    spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(test_spectra)

    # with added embededings
    model = ms2deepscore_model()
    correct_subsetted_set.add_embeddings(model)
    spectrum_set.add_embeddings(model)

    assert correct_subsetted_set == spectrum_set.subset_spectra([0,1,2,3,4])
