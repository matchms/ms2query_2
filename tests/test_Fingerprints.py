import numpy as np
import pytest
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi

from ms2query.benchmarking.SpectrumDataSet import Fingerprints
from tests.conftest import get_inchikey_inchi_pairs


@pytest.fixture
def dummy_fingerprints():
    inchikey_inchi_pairs = get_inchikey_inchi_pairs(5)
    inchi_per_inchikey = {compound_tuple[0][:14]: compound_tuple[1] for compound_tuple in inchikey_inchi_pairs}
    fingerprints = Fingerprints(inchi_per_inchikey, fingerprint_type="daylight", nbits=100)
    return fingerprints


def test_fingerprint_creation(dummy_fingerprints):
    assert dummy_fingerprints.fingerprints.shape== (7,100)
    for inchikey, inchi in list(dummy_fingerprints.most_common_inchi_per_inchikey.items()):
        fingerprint = _derive_fingerprint_from_inchi(inchi, fingerprint_type="daylight", nbits=100)
        np.testing.assert_array_equal(fingerprint, dummy_fingerprints.get_fingerprints([inchikey])[0])

def test_subsetting_fingerprint(dummy_fingerprints):
    selected_inchikeys = list(dummy_fingerprints.most_common_inchi_per_inchikey.keys())[-2:]
    subsetted_fingerprints = dummy_fingerprints.subset_fingerprints(selected_inchikeys)
    assert subsetted_fingerprints.fingerprints.shape== (2,100)
    assert list(subsetted_fingerprints.most_common_inchi_per_inchikey.keys()) == selected_inchikeys
    for inchikey in selected_inchikeys:
        fingerprint = _derive_fingerprint_from_inchi(dummy_fingerprints.most_common_inchi_per_inchikey[inchikey], fingerprint_type="daylight", nbits=100)
        np.testing.assert_array_equal(fingerprint, subsetted_fingerprints.get_fingerprints([inchikey])[0])

def test_add_fingerprint(dummy_fingerprints):
    inchikey_inchi_pairs = get_inchikey_inchi_pairs(7)[-2:]
    inchi_per_inchikey = {compound_tuple[0][:14]: compound_tuple[1] for compound_tuple in inchikey_inchi_pairs}
    dummy_fingerprints.add_new_inchikeys(inchi_per_inchikey)

    assert dummy_fingerprints.fingerprints.shape== (7,100)
    for inchikey, inchi in inchi_per_inchikey.items():
        fingerprint = _derive_fingerprint_from_inchi(inchi, fingerprint_type="daylight", nbits=100)
        np.testing.assert_array_equal(fingerprint, dummy_fingerprints.get_fingerprints([inchikey])[0])
