import numpy as np

from ms2query.benchmarking.SpectrumDataSet import SpectraWithMS2DeepScoreEmbeddings, SpectraWithFingerprints
from ms2query.benchmarking.reference_methods.predict_using_closest_tanimoto import (
    predict_using_closest_tanimoto, predict_using_closest_tanimoto_single_spectrum,
    get_average_predictions_for_closely_related_metabolites, get_inchikey_and_tanimoto_scores_for_top_k)
from tests.conftest import ms2deepscore_model, create_test_spectra
import pytest



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