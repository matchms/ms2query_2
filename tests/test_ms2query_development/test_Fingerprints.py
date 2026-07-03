import numpy as np
import pytest
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi
from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.Fingerprints import Fingerprints, get_similarity_matrix
from tests.helper_functions import create_test_spectra, get_inchikey_inchi_pairs


@pytest.fixture
def dummy_fingerprints() -> Fingerprints:
    return make_test_fingerprints(5)


def get_inchikey_inchi_dict(nr_of_inchikeys):
    inchikey_inchi_pairs = get_inchikey_inchi_pairs(nr_of_inchikeys)
    return {compound_tuple[0][:14]: compound_tuple[1] for compound_tuple in inchikey_inchi_pairs}


def make_test_fingerprints(nr_of_inchikeys=5):
    inchi_per_inchikey = get_inchikey_inchi_dict(nr_of_inchikeys)
    fingerprints = Fingerprints.compute_fingerprints_from_inchi(
        inchi_per_inchikey, fingerprint_type="daylight", nbits=2048
    )
    return fingerprints


def test_compute_fingerprints_from_inchi():
    nr_of_inchikeys = 5
    inchi_per_inchikey = get_inchikey_inchi_dict(nr_of_inchikeys)
    fingerprints = Fingerprints.compute_fingerprints_from_inchi(
        inchi_per_inchikey, fingerprint_type="daylight", nbits=2048
    )
    assert fingerprints.fingerprints.shape == (nr_of_inchikeys, 2048)
    for inchikey in fingerprints.inchikeys:
        fingerprint = _derive_fingerprint_from_inchi(
            inchi_per_inchikey[str(inchikey)], fingerprint_type="daylight", nbits=2048
        )
        np.testing.assert_array_equal(fingerprint, fingerprints.get_fingerprints([inchikey])[0])


def test_subsetting_fingerprint():
    inchi_per_inchikey = get_inchikey_inchi_dict(nr_of_inchikeys=5)
    fingerprints = Fingerprints.compute_fingerprints_from_inchi(
        inchi_per_inchikey, fingerprint_type="daylight", nbits=2048
    )
    inchi_per_inchikey_subset = get_inchikey_inchi_dict(nr_of_inchikeys=2)
    selected_inchikeys = list(inchi_per_inchikey_subset.keys())
    subsetted_fingerprints = fingerprints.subset_fingerprints(selected_inchikeys)
    assert subsetted_fingerprints.fingerprints.shape == (2, 2048)
    assert list(subsetted_fingerprints.inchikeys) == selected_inchikeys
    assert subsetted_fingerprints == Fingerprints.compute_fingerprints_from_inchi(
        inchi_per_inchikey_subset, fingerprint_type="daylight", nbits=2048
    )


def test_combine_fingerprint(dummy_fingerprints):
    inchikey_inchi_pairs = get_inchikey_inchi_pairs(7)[-2:]
    inchi_per_inchikey = {compound_tuple[0][:14]: compound_tuple[1] for compound_tuple in inchikey_inchi_pairs}
    last_two_fingerprints = Fingerprints.compute_fingerprints_from_inchi(
        inchi_per_inchikey, nbits=2048, fingerprint_type="daylight"
    )
    combined_fingerprints = Fingerprints.combine_fingerprints(dummy_fingerprints, last_two_fingerprints)

    assert combined_fingerprints.fingerprints.shape == (7, 2048)
    for inchikey, inchi in inchi_per_inchikey.items():
        fingerprint = _derive_fingerprint_from_inchi(inchi, fingerprint_type="daylight", nbits=2048)
        np.testing.assert_array_equal(fingerprint, combined_fingerprints.get_fingerprints([inchikey])[0])


def test_combine_fingerprints_with_replacing():
    inchikey_inchi_pairs_1 = {
        "RYYVLZVUVIJVGH": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
        "ZPUCINDJVBIVPJ": "InChI=1S/C17H21NO4/c1-18-12-8-9-13(18)15(17(20)21-2)14(10-12)22-16"
        "(19)11-6-4-3-5-7-11/h3-7,12-15H,8-10H2,1-2H3/t12-,13+,14-,15+/m0/s1",
    }
    inchikey_inchi_pairs_2 = {
        "RZVAJINKPMORJF": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
        "ZPUCINDJVBIVPJ": "InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)",
    }
    combined = {
        "RYYVLZVUVIJVGH": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
        "ZPUCINDJVBIVPJ": "InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)",
        "RZVAJINKPMORJF": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
    }
    fingerprints_1 = Fingerprints.compute_fingerprints_from_inchi(inchikey_inchi_pairs_1, "daylight", nbits=2048)
    fingerprints_2 = Fingerprints.compute_fingerprints_from_inchi(inchikey_inchi_pairs_2, "daylight", nbits=2048)
    correct_combined_fingerprints = Fingerprints.compute_fingerprints_from_inchi(combined, "daylight", nbits=2048)
    combined_fingerprints = Fingerprints.combine_fingerprints(fingerprints_1, fingerprints_2, allow_replacing=True)
    assert combined_fingerprints == correct_combined_fingerprints

    with pytest.raises(ValueError):
        Fingerprints.combine_fingerprints(fingerprints_1, fingerprints_2, allow_replacing=False)


def test_fingerprint_from_spectra():
    test_spectra = create_test_spectra(2, nr_of_inchikeys=3)
    spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(test_spectra)
    fingerprints = Fingerprints.from_spectrum_set(spectrum_set, "daylight", 2048)
    fingerprints_from_inchi = Fingerprints.compute_fingerprints_from_inchi(get_inchikey_inchi_dict(3), "daylight", 2048)
    assert fingerprints == fingerprints_from_inchi


def test_get_similarity_matrix():
    fingerprints_1 = make_test_fingerprints(5)
    fingerprints_2 = make_test_fingerprints(3)
    matrix = get_similarity_matrix(fingerprints_1, fingerprints_2)
    assert matrix.shape == (5, 3)
