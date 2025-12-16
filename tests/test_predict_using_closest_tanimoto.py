import numpy as np
import pytest
from ms2query.benchmarking.reference_methods.predict_using_closest_tanimoto import (
    get_average_predictions_for_closely_related_metabolites,
    get_inchikey_and_tanimoto_scores_for_top_k,
    predict_using_closest_tanimoto,
    predict_using_closest_tanimoto_single_spectrum,
    select_inchikeys_with_highest_ms2deepscore,
)
from ms2query.benchmarking.SpectrumDataSet import SpectraWithFingerprints, SpectraWithMS2DeepScoreEmbeddings
from tests.conftest import create_test_spectra, ms2deepscore_model


def test_predict_using_closest_tanimoto():
    """Only very basic test that the function runs and that the output is the right format"""
    model = ms2deepscore_model()
    library_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(nr_of_inchikeys=7), model)
    test_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(1, nr_of_inchikeys=3), model)
    predicted_inchikeys, scores = predict_using_closest_tanimoto(library_spectra, test_spectra, 3, 3)

    assert isinstance(predicted_inchikeys, list)
    assert len(predicted_inchikeys) == 3
    assert isinstance(scores, list)
    assert len(scores) == 3

def test_predict_using_closest_tanimoto_single_spectrum():
    """Only very basic test that the function runs and that the output is the right format"""
    model = ms2deepscore_model()
    library_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(nr_of_inchikeys=7), model)
    test_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(1, nr_of_inchikeys=1), model)
    predicted_inchikey, score = predict_using_closest_tanimoto_single_spectrum(library_spectra, test_spectra, 3, 3)

    assert isinstance(predicted_inchikey, str)
    assert len(predicted_inchikey) ==14
    assert isinstance(score, float)

def test_select_inchikeys_with_highest_ms2deepscore():
    test_spectra = create_test_spectra(nr_of_inchikeys=7)
    spectra = SpectraWithFingerprints(test_spectra)

    ms2deepscores = np.zeros(len(test_spectra))
    ms2deepscores[2] = 0.4
    ms2deepscores[5] = 0.9
    ms2deepscores[7] = 0.6
    inchikeys_with_highest_ms2deepscore = select_inchikeys_with_highest_ms2deepscore(spectra, ms2deepscores, 3)
    expected_inchikeys = list(spectra.spectrum_indexes_per_inchikey.keys())[:3]
    assert set(expected_inchikeys) == set(inchikeys_with_highest_ms2deepscore)
    print(inchikeys_with_highest_ms2deepscore)

def test_get_average_predictions_for_closely_related_metabolites():
    test_spectra = create_test_spectra(nr_of_inchikeys=7)
    # Select different number per inchikey (only one for the first) to check that it is correctly weighted.
    test_spectra = test_spectra.copy()[2:]
    spectra = SpectraWithFingerprints(test_spectra)

    inchikeys = list(spectra.inchikey_fingerprint_pairs.keys())[:3]
    ms2deepscores = np.zeros(len(spectra.spectra))
    ms2deepscores[0] = 0.8
    ms2deepscores[[1,2,3]] = 0.6
    ms2deepscores[4] = 0.6
    ms2deepscores[5] = 0.8
    ms2deepscores[6] = 0.7
    # the average per inchikey is 0.8, 0.6, 0.7, so average overall should be 0.7
    average_predicted_score = get_average_predictions_for_closely_related_metabolites(spectra,
                                                                                      inchikeys,
                                                                                      ms2deepscores)
    assert np.allclose(average_predicted_score, np.array(0.7), atol=1e-5)

@pytest.mark.parametrize(
    "k",
    [1, 3, 7],
)
def test_get_inchikey_and_tanimoto_scores_for_top_k(k):
    spectra = SpectraWithFingerprints(create_test_spectra(nr_of_inchikeys=7))
    inchikey = list(spectra.inchikey_fingerprint_pairs.keys())[2]

    top_inchikeys, tanimoto_scores_for_top_k = get_inchikey_and_tanimoto_scores_for_top_k(
        spectra, inchikey,k)

    assert inchikey in top_inchikeys
    assert len(top_inchikeys) == k
    assert len(tanimoto_scores_for_top_k) == k
    assert len(set(top_inchikeys)) == k
    assert tanimoto_scores_for_top_k[top_inchikeys.index(inchikey)] == 1.0, \
        "The exact match is expected to have a score of 1.0"