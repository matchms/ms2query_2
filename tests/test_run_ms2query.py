from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.TopKTanimotoScores import TopKTanimotoScores
from ms2query.run_ms2query import extract_metadata_from_library, run_ms2query
from tests.helper_functions import create_test_spectra, ms2deepscore_model


def test_run_ms2query():
    model = ms2deepscore_model()
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(create_test_spectra(nr_of_inchikeys=7))
    test_spectra = AnnotatedSpectrumSet.create_spectrum_set(create_test_spectra(1, nr_of_inchikeys=3))
    library_spectra.add_embeddings(model)
    test_spectra.add_embeddings(model)
    fingerprints = Fingerprints.from_spectrum_set(library_spectra, "daylight", 100)
    top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(fingerprints, fingerprints, 3)
    spectrum_indices_per_inchikey = library_spectra.spectrum_indices_per_inchikey
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
    results = run_ms2query(
        test_spectra.embeddings,
        library_spectra.embeddings,
        metadata_library,
        spectrum_indices_per_inchikey,
        top_k_tanimoto_scores,
    )
    print(results)
