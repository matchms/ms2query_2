from ms2query.benchmarking.Embeddings import Embeddings
from tests.conftest import create_test_spectra, ms2deepscore_model


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
