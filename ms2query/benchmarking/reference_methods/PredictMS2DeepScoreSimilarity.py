from typing import Tuple
import numpy as np
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm

from ms2query.benchmarking.Embeddings import Embeddings


def predict_top_ms2deepscores(
    library_embeddings: Embeddings, query_embeddings: Embeddings, batch_size: int = 500, k=1
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory efficient way of calculating the highest MS2DeepScores

    When doing large matrix multiplications the memory footprint of storing the output matrix can be large.
    E.g. when doing 500.000 vs 10.000 spectra this is a very large matrix. On a laptop this can result in very
    slow run times. If only the highest MS2DeepScore is needed processing in batches prevents using too much memory

    Args:
        library_embeddings: The embeddings of the library spectra
        query_embeddings: The embeddings of the query spectra
        batch_size: The number of query embeddings processed at the same time.
                    Setting a lower batch_size results in a lower memory footprint.
        k: Number of highest matches to return

    Returns:
        List[List[int]: indexes of highest scores and the value for the highest score.
        Per query embedding the top k highest indexes are given.
        List[List[float]: the highest scores.
    """
    top_indexes_per_batch = []
    top_scores_per_batch = []
    num_of_query_embeddings = query_embeddings.embeddings.shape[0]
    # loop over the batches
    for start_idx in tqdm(
        range(0, num_of_query_embeddings, batch_size),
        desc="Predicting highest ms2deepscore per batch of "
        + str(min(batch_size, num_of_query_embeddings))
        + " embeddings",
    ):
        end_idx = min(start_idx + batch_size, num_of_query_embeddings)
        selected_query_embeddings = query_embeddings.embeddings[start_idx:end_idx]
        score_matrix = cosine_similarity_matrix(selected_query_embeddings, library_embeddings.embeddings)
        top_n_idx = np.argsort(score_matrix, axis=1)[:, -k:][:, ::-1]
        top_n_scores = np.take_along_axis(score_matrix, top_n_idx, axis=1)
        top_indexes_per_batch.append(top_n_idx)
        top_scores_per_batch.append(top_n_scores)
    # todo refactor to use the Embeddings class and return spectrum hashes instead of indexes
    return np.vstack(top_indexes_per_batch), np.vstack(top_scores_per_batch)
