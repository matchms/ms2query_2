import numpy as np
import pytest
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.reference_methods.PredictMS2DeepScoreSimilarity import (
    predict_top_ms2deepscores,
    select_inchikeys_with_highest_ms2deepscore,
)
from tests.helper_functions import create_test_spectra, get_library_and_test_spectra_not_identical, ms2deepscore_model


@pytest.mark.parametrize(
    "method",
    [
        predict_top_ms2deepscores,
    ],
)
def test_predict_highest_ms2deepscore_similarity(method):
    ms2ds_model = ms2deepscore_model()
    test_spectra = create_test_spectra(1)
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(test_spectra)
    query_spectra = AnnotatedSpectrumSet.create_spectrum_set(test_spectra)
    library_spectra.add_embeddings(ms2ds_model)
    query_spectra.add_embeddings(ms2ds_model)
    number_of_analogues = 2

    indices, distances = method(library_spectra.embeddings, query_spectra.embeddings, k=number_of_analogues)
    assert indices.shape == (len(test_spectra), number_of_analogues)
    assert distances.shape == (len(test_spectra), number_of_analogues)
    for i, row in enumerate(indices):
        assert row[0] == i, "The highest predictions should be against itself"
        assert np.allclose(distances[i][0], 1.0, atol=1e-5)


def test_select_inchikeys_with_highest_ms2deepscore():
    library_spectra, query_spectra = get_library_and_test_spectra_not_identical()
    inschikeys_with_highest_scores = select_inchikeys_with_highest_ms2deepscore(query_spectra, library_spectra, 2)
    assert inschikeys_with_highest_scores == [
        ["RZVAJINKPMORJF", "RYYVLZVUVIJVGH"],
        ["RZVAJINKPMORJF", "ZPUCINDJVBIVPJ"],
        ["RYYVLZVUVIJVGH", "RZVAJINKPMORJF"],
    ]
