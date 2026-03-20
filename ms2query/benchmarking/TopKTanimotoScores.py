from pathlib import Path
import numpy as np
import pandas as pd
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.metrics import generalized_tanimoto_similarity_matrix


class TopKTanimotoScores:
    def __init__(
        self, tanimoto_scores_for_top_k: np.ndarray, top_k_inchikeys: np.ndarray, inchikey_indexes: np.ndarray
    ):
        """Stores the top k scores between two lists of inchikeys"""

        self.k = tanimoto_scores_for_top_k.shape[1]
        self.top_k_inchikeys_and_scores: pd.DataFrame = self._create_multi_index(
            tanimoto_scores_for_top_k, top_k_inchikeys, inchikey_indexes
        )

    def _create_multi_index(
        self, tanimoto_scores_for_top_k: np.ndarray, top_k_inchikeys: np.ndarray, inchikey_indexes: np.ndarray
    ):
        """Creates a pandas dataframe with multi index.
        For each inchikey the top ten highest inchikeys and the corresponding score are stored."""
        columns = pd.MultiIndex.from_product(
            [[f"Rank_{i + 1}" for i in range(self.k)], ["inchikey", "score"]], names=["result_rank", "attribute"]
        )

        combined_data = np.empty((len(inchikey_indexes), self.k * 2), dtype=object)
        combined_data[:, 0::2] = top_k_inchikeys
        combined_data[:, 1::2] = tanimoto_scores_for_top_k
        df = pd.DataFrame(combined_data, index=inchikey_indexes, columns=columns)

        # Cast score columns to float64
        score_cols = [(rank, "score") for rank in [f"Rank_{i + 1}" for i in range(self.k)]]
        df[score_cols] = df[score_cols].astype(float)

        return df

    @classmethod
    def calculate_from_fingerprints(cls, query_fingerprints: Fingerprints, target_fingerprints: Fingerprints, k):
        """
        Gets the top k highest inchikeys and scores for each inchikey in query_fingerprints from target_fingerprints
        """
        if target_fingerprints.fingerprints.shape[0] < k:
            raise ValueError("K cannot be larger than the number of fingerprints")
        similarity_scores = generalized_tanimoto_similarity_matrix(
            query_fingerprints.fingerprints, target_fingerprints.fingerprints
        )
        inchikey_indexes_of_top_k = np.argpartition(similarity_scores, -k, axis=1)[:, -k:]
        top_k_inchikeys = np.array(target_fingerprints.inchikeys)[inchikey_indexes_of_top_k]
        tanimoto_scores_for_top_k = similarity_scores[
            np.arange(similarity_scores.shape[0])[:, None], inchikey_indexes_of_top_k
        ]
        return cls(tanimoto_scores_for_top_k, top_k_inchikeys, np.array(query_fingerprints.inchikeys))

    def select_top_k_inchikeys_and_scores(self, inchikey) -> dict[str, float]:
        """Returns a dictionary with inchikeys and scores for the given inchikey"""
        # Select row with results
        top_k_inchikeys_and_scores = self.top_k_inchikeys_and_scores.loc[inchikey]
        # Unstack the multi index to get a dataframe with inchikeys and scores and index rank_1, rank_2 etc.
        top_k_inchikeys_and_scores = top_k_inchikeys_and_scores.unstack(level="result_rank").T
        # Convert to dictionary with inchikey, score pairs.
        return dict(zip(top_k_inchikeys_and_scores["inchikey"], top_k_inchikeys_and_scores["score"]))

    def select_top_k_inchikeys(self, inchikey) -> list[str]:
        return list(self.top_k_inchikeys_and_scores.loc[inchikey].xs("inchikey", level="attribute"))

    def select_average_score(self, inchikey) -> float:
        """Returns the average tanimoto score for the top k highest scores for the specified inchikey"""
        return float(self.top_k_inchikeys_and_scores.loc[inchikey].xs("score", level="attribute").mean())

    def get_all_average_tanimoto_scores(self) -> dict[str, float]:
        """Returns all average tanimoto scores for the top k per inchikey"""
        # Get the scores
        scores_df: pd.DataFrame = self.top_k_inchikeys_and_scores.xs("score", axis=1, level="attribute")  # type: ignore

        average_per_inchikey_df = scores_df.mean(axis=1)
        return average_per_inchikey_df.to_dict()

    def save(self, path: str | Path) -> None:
        """Save the TopKTanimotoScores to disk as a parquet file.

        Args:
            path: File path without extension, e.g. "/data/top_k_scores".
        """
        Path(path).with_suffix(".parquet").parent.mkdir(parents=True, exist_ok=True)
        self.top_k_inchikeys_and_scores.to_parquet(Path(path).with_suffix(".parquet"))

    @classmethod
    def load(cls, path: str | Path) -> "TopKTanimotoScores":
        """Load a previously saved TopKTanimotoScores from disk.

        Args:
            path: File path without extension, e.g. "/data/top_k_scores".

        Returns:
            A fully reconstructed TopKTanimotoScores instance.
        """
        df = pd.read_parquet(Path(path).with_suffix(".parquet"))
        df.columns.names = ["result_rank", "attribute"]

        instance = cls.__new__(cls)
        instance.k = len(df.columns.get_level_values("result_rank").unique())
        instance.top_k_inchikeys_and_scores = df
        return instance
