import pytest
from ms2query.benchmarking.Embeddings import Embeddings, calculate_ms2deepscore_df
from tests.helper_functions import create_test_spectra, get_library_and_test_spectra_not_identical, ms2deepscore_model


def test_subset_embeddings():
    test_spectra = create_test_spectra()
    model = ms2deepscore_model()
    embeddings = Embeddings.create_from_spectra(test_spectra, model)
    subset = embeddings.subset_embeddings(test_spectra[:4])
    correct_subset = Embeddings.create_from_spectra(test_spectra[:4], model)
    assert subset == correct_subset
    selected_disordered_spectra = [test_spectra[i] for i in [7, 2, 3]]
    subset = embeddings.subset_embeddings(selected_disordered_spectra)
    correct_subset = Embeddings.create_from_spectra(selected_disordered_spectra, model)
    assert subset == correct_subset
    with pytest.raises(ValueError):
        embeddings = Embeddings.create_from_spectra(test_spectra[:4], model)
        embeddings.subset_embeddings(test_spectra)


def test_add_embeddings():
    test_spectra = create_test_spectra()
    model = ms2deepscore_model()
    embeddings_1 = Embeddings.create_from_spectra(test_spectra[:4], model)
    embeddings_2 = Embeddings.create_from_spectra(test_spectra[4:], model)
    correct_combined_embeddings = Embeddings.create_from_spectra(test_spectra, model)
    combined_embeddings = embeddings_1 + embeddings_2
    assert combined_embeddings == correct_combined_embeddings
    with pytest.raises(ValueError):
        _ = embeddings_1 + embeddings_1


def test_calculate_ms2deepscore_df():
    library_spectra, query_spectra = get_library_and_test_spectra_not_identical()
    calculate_ms2deepscore_df(library_spectra.embeddings, query_spectra.embeddings)
