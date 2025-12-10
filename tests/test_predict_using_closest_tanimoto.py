import numpy as np

from ms2query.benchmarking.SpectrumDataSet import SpectraWithMS2DeepScoreEmbeddings, SpectraWithFingerprints
from ms2query.benchmarking.reference_methods.predict_using_closest_tanimoto import (
    predict_using_closest_tanimoto, predict_using_closest_tanimoto_single_spectrum,
    get_average_predictions_for_closely_related_metabolites, get_inchikey_and_tanimoto_scores_for_top_k)
from tests.conftest import ms2deepscore_model, create_test_spectra
import pytest


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