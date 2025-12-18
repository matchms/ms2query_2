import numpy as np
import pandas as pd

from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.metrics import generalized_tanimoto_similarity_matrix


class TopKTanimotoScores:
    def __init__(self, tanimoto_scores_for_top_k: np.ndarray,
                 top_k_inchikeys: np.ndarray,
                 inchikey_indexes: np.ndarray):

        self.k = tanimoto_scores_for_top_k.shape[1]
        self.top_k_inchikeys_and_scores = self.create_multi_index(tanimoto_scores_for_top_k,
                                                                  top_k_inchikeys,
                                                                  inchikey_indexes)

    def create_multi_index(self,  tanimoto_scores_for_top_k: np.ndarray,
                 top_k_inchikeys: np.ndarray,
                 inchikey_indexes: np.ndarray):
        columns = pd.MultiIndex.from_product([[f"Rank_{i + 1}" for i in range(self.k)], ["inchikey", "score"]],
            names=["result_rank", "attribute"]
        )

        combined_data = np.empty((len(inchikey_indexes), self.k * 2), dtype=object)
        combined_data[:, 0::2] = top_k_inchikeys
        combined_data[:, 1::2] = tanimoto_scores_for_top_k
        return pd.DataFrame(combined_data, index=inchikey_indexes, columns=columns)


    @classmethod
    def calculate_from_fingerprints(cls, query_fingerprints: Fingerprints, target_fingerprints: Fingerprints, k):
        """Gets the top k highest inchikeys and scores for each inchikey in query_fingerprints from target_fingerprints"""
        similarity_scores = generalized_tanimoto_similarity_matrix(query_fingerprints.fingerprints, target_fingerprints.fingerprints)
        inchikey_indexes_of_top_k = np.argpartition(similarity_scores, -k, axis=1)[:, -k:]
        top_k_inchikeys = target_fingerprints.index_to_inchikey[inchikey_indexes_of_top_k]
        tanimoto_scores_for_top_k = similarity_scores[np.arange(similarity_scores.shape[0])[:, None], inchikey_indexes_of_top_k]
        return cls(tanimoto_scores_for_top_k, top_k_inchikeys, query_fingerprints.index_to_inchikey)

    def select_top_k_inchikeys_and_scores(self, inchikey):
        """Returns a DF with inchikeys and scores"""
        return self.top_k_inchikeys_and_scores.loc[inchikey].unstack(level='result_rank').T

    def select_top_k_inchikeys(self, inchikey):
        return self.top_k_inchikeys_and_scores.loc[inchikey].xs('inchikey', level='attribute')

    def select_average_score(self, inchikey):
        return self.top_k_inchikeys_and_scores.loc[inchikey].xs('score', level='attribute').mean()

    def get_all_average_tanimoto_scores(self):
        return self.top_k_inchikeys_and_scores.xs('score', axis=1, level='attribute').mean(axis=1)
