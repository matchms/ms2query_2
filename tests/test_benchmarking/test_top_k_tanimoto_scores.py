import numpy as np
import pandas as pd
import pytest
from ms2query.benchmarking.TopKTanimotoScores import TopKTanimotoScores
from tests.helper_functions import make_test_fingerprints


def test_methods_top_k_tanimoto_scores():
    scores = np.array([[1.0, 0.8], [0.9, 0.7], [0.8, 0.8]])
    inchikeys_in_top_2 = np.array([["A", "B"], ["B", "C"], ["C", "A"]])
    inchikeys = np.array(["A", "B", "C"])
    top_scores = TopKTanimotoScores(scores, inchikeys_in_top_2, inchikeys)

    assert top_scores.select_top_k_inchikeys_and_scores("A") == {"A": 1.0, "B": 0.8}

    assert top_scores.select_top_k_inchikeys("A") == ["A", "B"]

    assert top_scores.select_average_score("A") == pytest.approx(0.9)

    assert top_scores.get_all_average_tanimoto_scores() == {"A": 0.9, "B": 0.8, "C": 0.8}


def test_calculate_from_fingerprints():
    fingerprints = make_test_fingerprints(nbits=5, nr_of_inchikeys=5)
    top_scores = TopKTanimotoScores.calculate_from_fingerprints(fingerprints, fingerprints, 2)
    assert top_scores.select_top_k_inchikeys_and_scores("AAAAAAAAAAAAAE") == {
        "AAAAAAAAAAAAAD": 0.75,
        "AAAAAAAAAAAAAE": 1.0,
    }


@pytest.fixture
def sample_scores():
    """Creates a simple TopKTanimotoScores instance for testing."""
    tanimoto_scores = np.array(
        [
            [0.9, 0.7, 0.5],
            [0.8, 0.6, 0.4],
            [0.95, 0.85, 0.75],
        ]
    )
    top_k_inchikeys = np.array(
        [
            ["INCHI_A", "INCHI_B", "INCHI_C"],
            ["INCHI_B", "INCHI_C", "INCHI_A"],
            ["INCHI_C", "INCHI_A", "INCHI_B"],
        ]
    )
    inchikey_indexes = np.array(["QUERY_1", "QUERY_2", "QUERY_3"])
    return TopKTanimotoScores(tanimoto_scores, top_k_inchikeys, inchikey_indexes)


# ----- save and load tests -----
def test_save_creates_parquet_file(sample_scores, tmp_path):
    sample_scores.save(tmp_path / "test_scores")
    assert (tmp_path / "test_scores.parquet").exists()


def test_save_creates_parent_directories(sample_scores, tmp_path):
    sample_scores.save(tmp_path / "nested" / "dir" / "test_scores")
    assert (tmp_path / "nested" / "dir" / "test_scores.parquet").exists()


def test_roundtrip_produces_identical_object(sample_scores, tmp_path):
    sample_scores.save(tmp_path / "test_scores")
    loaded = TopKTanimotoScores.load(tmp_path / "test_scores")

    assert loaded.k == sample_scores.k
    pd.testing.assert_frame_equal(loaded.top_k_inchikeys_and_scores, sample_scores.top_k_inchikeys_and_scores)
    assert sample_scores.select_top_k_inchikeys_and_scores("QUERY_1") == loaded.select_top_k_inchikeys_and_scores(
        "QUERY_1"
    )
    assert sample_scores.select_top_k_inchikeys("QUERY_2") == loaded.select_top_k_inchikeys("QUERY_2")
    assert sample_scores.select_average_score("QUERY_3") == pytest.approx(loaded.select_average_score("QUERY_3"))


def test_roundtrip_accepts_string_path(sample_scores, tmp_path):
    sample_scores.save(str(tmp_path / "test_scores"))
    loaded = TopKTanimotoScores.load(str(tmp_path / "test_scores"))
    pd.testing.assert_frame_equal(loaded.top_k_inchikeys_and_scores, sample_scores.top_k_inchikeys_and_scores)
