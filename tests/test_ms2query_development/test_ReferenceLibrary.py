import os
import pandas as pd
from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.Fingerprints import Fingerprints
from ms2query.ms2query_development.ReferenceLibrary import (
    ReferenceLibrary,
    extract_metadata_from_library,
)
from ms2query.ms2query_development.TopKTanimotoScores import TopKTanimotoScores
from tests.helper_functions import TEST_RESOURCES_PATH, create_test_spectra, ms2deepscore_model


def test_run_ms2query():
    model = ms2deepscore_model()
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(create_test_spectra(nr_of_inchikeys=7))
    test_spectra = create_test_spectra(1, nr_of_inchikeys=3)
    library_spectra.add_embeddings(model)
    fingerprints = Fingerprints.from_spectrum_set(library_spectra, "daylight", 100)
    top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(fingerprints, fingerprints, 3)
    metadata_library = extract_metadata_from_library(
        library_spectra,
        [
            "precursor_mz",
            "collision_energy",
            "compound_name",
            "smiles",
            "inchikey",
        ],
    )

    results = ReferenceLibrary(model, library_spectra.embeddings, top_k_tanimoto_scores, metadata_library).run_ms2query(
        test_spectra
    )
    print(results)


def test_create_library(tmp_path):
    lib_spectra = create_test_spectra(nr_of_inchikeys=10, number_of_spectra_per_inchikey=3)
    # save_as_mgf(lib_spectra, os.path.join(tmp_path, "library_spectra.mgf"))
    ms2deepscore_model_file = os.path.join(TEST_RESOURCES_PATH, "ms2deepscore_testmodel_v1.pt")
    library = ReferenceLibrary.create_from_spectra(lib_spectra, ms2deepscore_model_file)
    library.save(tmp_path)
    assert (tmp_path / ReferenceLibrary.embedding_file_name).exists()
    assert (tmp_path / ReferenceLibrary.top_k_tanimoto_scores_file_name).exists()
    assert (tmp_path / ReferenceLibrary.reference_metadata_file_name).exists()


def test_create_and_use_library(tmp_path):
    lib_spectra = create_test_spectra(nr_of_inchikeys=10, number_of_spectra_per_inchikey=3)
    ms2deepscore_model_file = os.path.join(TEST_RESOURCES_PATH, "ms2deepscore_testmodel_v1.pt")
    ms2query_library = ReferenceLibrary.create_from_spectra(lib_spectra, ms2deepscore_model_file)
    ms2query_library.save(tmp_path)
    test_spectra = create_test_spectra(1, nr_of_inchikeys=3)
    results = ms2query_library.run_ms2query(test_spectra)

    ms2query_library_2 = ReferenceLibrary.load_from_files(
        ms2deepscore_model_file,
        tmp_path / ReferenceLibrary.embedding_file_name,
        tmp_path / ReferenceLibrary.top_k_tanimoto_scores_file_name,
        tmp_path / ReferenceLibrary.reference_metadata_file_name,
    )

    results_2 = ms2query_library_2.run_ms2query(test_spectra)
    pd.testing.assert_frame_equal(results, results_2)


def test_add_spectra():
    lib_spectra = create_test_spectra(nr_of_inchikeys=10, number_of_spectra_per_inchikey=3)
    first_spectra = lib_spectra[:25]
    later_spectra = lib_spectra[25:]
    ms2deepscore_model_file = os.path.join(TEST_RESOURCES_PATH, "ms2deepscore_testmodel_v1.pt")
    ms2query_library = ReferenceLibrary.create_from_spectra(first_spectra, ms2deepscore_model_file)
    ms2query_library.add_spectra(later_spectra)

    ms2query_library_2 = ReferenceLibrary.create_from_spectra(lib_spectra, ms2deepscore_model_file)

    assert ms2query_library.reference_embeddings == ms2query_library_2.reference_embeddings
    assert ms2query_library.top_k_tanimoto_scores == ms2query_library_2.top_k_tanimoto_scores
    pd.testing.assert_frame_equal(ms2query_library.reference_metadata, ms2query_library_2.reference_metadata)
    test_spectra = create_test_spectra(1, nr_of_inchikeys=3)

    results = ms2query_library.run_ms2query(test_spectra)
    results_2 = ms2query_library_2.run_ms2query(test_spectra)

    pd.testing.assert_frame_equal(results, results_2)


def test_run_semi_targeted_search():
    lib_spectra = create_test_spectra(nr_of_inchikeys=10, number_of_spectra_per_inchikey=3)
    ms2deepscore_model_file = os.path.join(TEST_RESOURCES_PATH, "ms2deepscore_testmodel_v1.pt")
    library = ReferenceLibrary.create_from_spectra(lib_spectra, ms2deepscore_model_file)
    test_spectra = create_test_spectra(1, nr_of_inchikeys=3)
    inchikeys = {spectrum.get("inchikey")[:14] for spectrum in lib_spectra}
    results = library.run_semi_targeted_ms2query(test_spectra, inchikeys)

    results_2 = library.run_ms2query(test_spectra)
    print(results)
    print(results_2)
    pd.testing.assert_frame_equal(results, results_2)
