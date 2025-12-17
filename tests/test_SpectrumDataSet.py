import numpy as np
import pytest
from ms2query.benchmarking.SpectrumDataSet import (
    SpectraWithFingerprints,
    SpectraWithMS2DeepScoreEmbeddings,
    SpectrumSet,
)
from tests.conftest import create_test_spectra, get_inchikey_inchi_pairs, ms2deepscore_model


@pytest.mark.parametrize(
    "library",
    [
        SpectrumSet(create_test_spectra()),
        SpectraWithFingerprints(create_test_spectra()),
        SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(), ms2deepscore_model()),
    ],
)
def test_spectrum_set_base(library):
    """Test all base functionality of SpectrumSet is implemented correctly
    also for all classes inheriting from it"""
    # test correct init
    assert len(library.spectra) == 9
    assert len(library.spectrum_indexes_per_inchikey) == 3
    assert sum(len(v) for v in library.spectrum_indexes_per_inchikey.values()) == 9

    # test correct copying
    new_copy = library.copy()
    assert len(new_copy.spectra) == 9
    assert len(new_copy.spectrum_indexes_per_inchikey) == 3
    assert sum(len(v) for v in new_copy.spectrum_indexes_per_inchikey.values()) == 9

    # test correctly adding spectra
    new_copy.add_spectra(library)
    new_number_of_spectra = 9 + 9
    assert len(new_copy.spectra) == new_number_of_spectra
    assert len(new_copy.spectrum_indexes_per_inchikey) == 3
    assert sum(len(v) for v in new_copy.spectrum_indexes_per_inchikey.values()) == new_number_of_spectra

    # test the original is not edited when adding spectra
    assert len(library.spectra) == 9
    assert len(library.spectrum_indexes_per_inchikey) == 3
    assert sum(len(v) for v in library.spectrum_indexes_per_inchikey.values()) == 9

    # test correct subsetting
    subset_indexes = [1, 4, 6, 7]
    subset = library.subset_spectra(subset_indexes)
    assert len(subset.spectra) == len(subset_indexes)
    assert len(subset.spectrum_indexes_per_inchikey) == 3
    assert sum(len(v) for v in subset.spectrum_indexes_per_inchikey.values()) == len(subset_indexes)
    assert isinstance(subset, library.__class__)


@pytest.mark.parametrize(
    "library",
    [
        SpectraWithFingerprints(create_test_spectra()),
        SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(), ms2deepscore_model()),
    ],
)
def test_spectra_with_fingerprints(library):
    """Test all functionality added in SpectraWithFingerprints also for all classes inheriting from it"""
    # test correct init
    assert len(library.inchikey_fingerprint_pairs) == 3

    # test correct copying
    new_copy = library.copy()
    assert len(new_copy.inchikey_fingerprint_pairs) == 3

    # test correctly adding inchikey_fingerprint_pairs when runnning add_spectra
    for inchikey_inchi_pairs, expected_nr_of_inchikeys in (
        (get_inchikey_inchi_pairs(5)[2:], 5),  # Some overlap
        (get_inchikey_inchi_pairs(5)[3:], 5),  # No overlap
        (get_inchikey_inchi_pairs(3), 3),  # Fully overlapping
        (get_inchikey_inchi_pairs(1), 3),  # Fully overlapping (but not all)
    ):
        spectra_to_add = SpectrumSet(create_test_spectra(2, inchikey_inchi_pairs=inchikey_inchi_pairs))
        new_copy = library.copy()
        new_copy.add_spectra(spectra_to_add)
        assert len(new_copy.inchikey_fingerprint_pairs) == expected_nr_of_inchikeys
        for inchikey in library.inchikey_fingerprint_pairs:
            assert np.array_equal(
                new_copy.inchikey_fingerprint_pairs[inchikey], library.inchikey_fingerprint_pairs[inchikey]
            )

    # test the original is not edited when adding spectra
    assert len(library.inchikey_fingerprint_pairs) == 3
    assert all(
        np.array_equal(library.inchikey_fingerprint_pairs[key], value)
        for key, value in SpectraWithFingerprints(create_test_spectra()).inchikey_fingerprint_pairs.items()
    )

    # test correct subsetting
    subset_indexes = [1, 6, 7]
    subset = library.subset_spectra(subset_indexes)
    assert len(subset.inchikey_fingerprint_pairs) == 2
    assert all(
        np.array_equal(library.inchikey_fingerprint_pairs[key], value)
        for key, value in subset.inchikey_fingerprint_pairs.items()
    )
    assert hasattr(subset, "update_fingerprint_per_inchikey")


def test_spectra_with_embeddings():
    library = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(), ms2deepscore_model())
    # test correct init
    assert library.embeddings.shape == (9, 100)

    # test correct copying
    new_copy = library.copy()
    assert new_copy.embeddings.shape == (9, 100)

    # test correctly adding spectra
    new_spectra = SpectraWithMS2DeepScoreEmbeddings(create_test_spectra(1), ms2deepscore_model())
    new_copy.add_spectra(new_spectra)
    assert new_copy.embeddings.shape == (12, 100)

    # test the original is not edited when adding spectra
    assert library.embeddings.shape == (9, 100)

    # test correct subsetting
    subset_indexes = [1, 4, 6, 7]
    subset = library.subset_spectra(subset_indexes)
    assert subset.embeddings.shape == (len(subset_indexes), 100)
    for i, index in enumerate(subset_indexes):
        assert np.all(library.embeddings[index] == subset.embeddings[i])

    # Check that subsetting on subset works. To make sure that a subset does not become of type SpectrumSet
    subsetted_subset = subset.subset_spectra([0, 1])
    assert subsetted_subset.embeddings.shape == (2, 100)
