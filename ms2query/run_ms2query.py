import json
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.importing import load_spectra
from ms2deepscore.models import load_model
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Embeddings import Embeddings
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.TopKTanimotoScores import TopKTanimotoScores


def run_ms2query(
    query_embeddings: Embeddings,
    library_embeddings: Embeddings,
    library_metadata: pd.DataFrame,
    spectrum_indices_per_inchikey: dict[str, Tuple[int, ...]],
    top_k_tanimoto_scores: TopKTanimotoScores,
    batch_size: int = 1000,
):
    num_of_query_embeddings = query_embeddings.embeddings.shape[0]

    library_index_highest_ms2deepscore = np.zeros((num_of_query_embeddings), dtype=int)
    ms2query_scores = []
    for start_idx in tqdm(
        range(0, num_of_query_embeddings, batch_size),
        desc="Predicting highest ms2deepscore per batch of "
        + str(min(batch_size, num_of_query_embeddings))
        + " embeddings",
    ):
        # Do MS2DeepScore predictions for batch
        end_idx = min(start_idx + batch_size, num_of_query_embeddings)
        selected_query_embeddings = query_embeddings.embeddings[start_idx:end_idx]
        score_matrix = cosine_similarity_matrix(selected_query_embeddings, library_embeddings.embeddings)
        highest_score_idx = np.argmax(score_matrix, axis=1)
        library_index_highest_ms2deepscore[start_idx:end_idx] = highest_score_idx

        # get predicted inchikeys
        predicted_inchikeys = library_metadata.iloc[highest_score_idx]["inchikey"]
        # Compute MS2Query reliability score
        ms2query_scores.extend(
            get_ms2query_reliability_prediction(
                predicted_inchikeys, spectrum_indices_per_inchikey, top_k_tanimoto_scores, score_matrix
            )
        )

    # construct results df
    results = library_metadata.iloc[library_index_highest_ms2deepscore]
    results["ms2query_reliability_prediction"] = ms2query_scores
    return results


def get_ms2query_reliability_prediction(
    predicted_inchikeys: list[str],
    spectrum_indices_per_inchikey,
    top_k_tanimoto_scores: TopKTanimotoScores,
    ms2deepscore_score_matrix,
) -> list[float]:
    ms2query_scores = []
    for query_spectrum_index, library_inchikey in enumerate(predicted_inchikeys):
        top_k_inchikeys = top_k_tanimoto_scores.select_top_k_inchikeys(library_inchikey[:14])
        maximum_ms2deepscores = np.zeros(top_k_tanimoto_scores.k, dtype=float)
        for i, inchikey in enumerate(top_k_inchikeys):
            spectrum_indexes = spectrum_indices_per_inchikey[inchikey]
            highest_ms2deepscore = np.max(ms2deepscore_score_matrix[query_spectrum_index, spectrum_indexes])
            maximum_ms2deepscores[i] = highest_ms2deepscore
        ms2query_scores.append(np.mean(maximum_ms2deepscores))
    # todo get the spectrum hashes instead of the indexes for lookup later.
    return ms2query_scores


def create_ms2query_library(library_spectra_file: str, ms2deepscore_model_file_name: str):
    """Loads in a library and saves the embeddings and top_k_tanimoto_scores"""
    spectrum_file_directory = Path("/some/dir/file.txt").parent
    embedding_file_location = spectrum_file_directory / "embeddings.npz"
    top_k_tanimoto_score_file_location = spectrum_file_directory / "top_k_tanimoto_scores.parquet"
    reference_metadata_file = spectrum_file_directory / "library_metadata.parquet"
    if embedding_file_location.exists():
        raise FileExistsError("There is already an embedding.npy file in the directory of your library spectra")
    if top_k_tanimoto_score_file_location.exists():
        raise FileExistsError(
            "There is already an top_k_tanimoto_scores.parquet file in the directory of your library spectra"
        )

    library_spectra = list(tqdm(load_spectra(library_spectra_file), "Loading library spectra"))
    library_spectra = AnnotatedSpectrumSet.create_spectrum_set(library_spectra)
    ms2deepscore_model = load_model(ms2deepscore_model_file_name)
    library_spectra.add_embeddings(ms2deepscore_model)

    library_spectra._embeddings.save(embedding_file_location)

    fingerprints = Fingerprints.from_spectrum_set(library_spectra, "daylight", 4096)
    top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
        fingerprints,
        fingerprints,
        k=8,
    )
    top_k_tanimoto_scores.save(top_k_tanimoto_score_file_location)
    reference_metadata = extract_metadata_from_library(
        library_spectra,
        [
            "precursor_mz",
            "retention_time",
            "collision_energy",
            "compound_name",
            "smiles",
            "inchikey",
        ],
    )
    reference_metadata.to_parquet(reference_metadata_file)


def extract_metadata_from_library(spectra: AnnotatedSpectrumSet, metadata_to_collect: list):
    collected_metadata = {key: [] for key in metadata_to_collect}
    collected_metadata["spectrum_hashes"] = []
    for spectrum in tqdm(spectra.spectra, desc="Extracting metadata df from spectra"):
        for metadata_key in metadata_to_collect:
            collected_metadata[metadata_key].append(spectrum.get(metadata_key))
        collected_metadata["spectrum_hashes"].append(spectrum.__hash__())
    return pd.DataFrame(collected_metadata)
