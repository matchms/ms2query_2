import numpy as np
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
    fingerprints = make_test_fingerprints(20)
    TopKTanimotoScores.calculate_from_fingerprints(fingerprints, fingerprints, 10)
