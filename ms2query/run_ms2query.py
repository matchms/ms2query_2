from collections import defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from matchms.importing import load_spectra
from matchms.Spectrum import Spectrum
from ms2deepscore.models import SiameseSpectralModel, load_model
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Embeddings import Embeddings, _to_json_serializable
from ms2query.benchmarking.Fingerprints import Fingerprints
from ms2query.benchmarking.TopKTanimotoScores import TopKTanimotoScores


class MS2QueryLibrary:
    # Set default file names to enable save and load per library
    embedding_file_name = "embeddings.npz"
    top_k_tanimoto_scores_file_name = "top_k_tanimoto_scores.parquet"
    reference_metadata_file_name = "library_metadata.parquet"
    ms2deepscore_model_file_name = "ms2deepscore_model.pt"
    metadata_to_store = [
        "precursor_mz",
        "retention_time",
        "collision_energy",
        "compound_name",
        "smiles",
        "inchikey",
    ]
    fingerprint_type = "daylight"
    fingerprint_nbits = 4096
    top_k_inchikeys = 8

    def __init__(
        self,
        ms2deepscore_model: SiameseSpectralModel,
        reference_embeddings: Embeddings,
        top_k_tanimoto_scores: TopKTanimotoScores,
        reference_metadata: pd.DataFrame,
    ):
        self.ms2deepscore_model = ms2deepscore_model
        self.reference_embeddings = reference_embeddings
        self.top_k_tanimoto_scores = top_k_tanimoto_scores
        self.reference_metadata = reference_metadata

        # Check that the loaded files match
        if _to_json_serializable(ms2deepscore_model.model_settings.get_dict()) != reference_embeddings.model_settings:
            raise ValueError(
                "The settings of the ms2deepscore model does not match the model used for creating the library embeddings"
            )
        if list(self.reference_metadata["spectrum_hashes"]) != [
            str(spectrum_hash) for spectrum_hash in reference_embeddings.index_to_spectrum_hash
        ]:
            raise ValueError("The loaded metadata does not match the used embeddings")
        if {inchikey[:14] for inchikey in reference_metadata["inchikey"]} != set(
            top_k_tanimoto_scores.top_k_inchikeys_and_scores.index
        ):
            raise ValueError("The inchikeys in the metadata and in the top_k_tanimoto_scores do not match")

        # Get the spectrum_indices_per_inchikey
        self.spectrum_indices_per_inchikey = defaultdict(list)
        for lib_spec_index, inchikey in enumerate(reference_metadata["inchikey"]):
            self.spectrum_indices_per_inchikey[inchikey[:14]].append(lib_spec_index)

    @classmethod
    def load_from_directory(cls, library_file_directory) -> "MS2QueryLibrary":
        reference_embeddings_file = library_file_directory / cls.embedding_file_name
        top_k_tanimoto_scores_file = library_file_directory / cls.top_k_tanimoto_scores_file_name
        reference_metadata_file = library_file_directory / cls.reference_metadata_file_name
        ms2deepscore_model_file_name = library_file_directory = cls.ms2deepscore_model_file_name
        return cls.load_from_files(
            ms2deepscore_model_file_name, reference_embeddings_file, top_k_tanimoto_scores_file, reference_metadata_file
        )

    @classmethod
    def load_from_files(
        cls,
        ms2deepscore_model_file_name,
        reference_embeddings_file,
        top_k_tanimoto_scores_file,
        reference_metadata_file,
    ) -> "MS2QueryLibrary":
        return cls(
            load_model(ms2deepscore_model_file_name),
            Embeddings.load(reference_embeddings_file),
            TopKTanimotoScores.load(top_k_tanimoto_scores_file),
            pd.read_parquet(reference_metadata_file),
        )

    @classmethod
    def create_from_spectra(
        cls,
        library_spectra: Sequence[Spectrum],
        ms2deepscore_model_file_name: str,
        store_file_directory=None,
        store_files=True,
    ) -> "MS2QueryLibrary":
        """Creates all the files needed for MS2Query and stores them"""
        if store_file_directory is None:
            store_file_directory = Path(ms2deepscore_model_file_name).parent
        if store_files:
            # Check the files don't exist yet
            for file in (
                store_file_directory / cls.embedding_file_name,
                store_file_directory / cls.top_k_tanimoto_scores_file_name,
                store_file_directory / cls.reference_metadata_file_name,
            ):
                if file.exists():
                    raise FileExistsError(f"There is already a file stored with the name {file}")

        # library_spectra = list(tqdm(load_spectra(library_spectra_file), "Loading library spectra"))
        library_spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(library_spectra)
        ms2deepscore_model = load_model(ms2deepscore_model_file_name)
        library_spectrum_set.add_embeddings(ms2deepscore_model)

        fingerprints = Fingerprints.from_spectrum_set(library_spectrum_set, cls.fingerprint_type, cls.fingerprint_nbits)
        top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
            fingerprints, fingerprints, cls.top_k_inchikeys
        )
        reference_metadata = extract_metadata_from_library(
            library_spectrum_set,
            cls.metadata_to_store,
        )

        if store_files:
            reference_metadata.to_parquet(store_file_directory / cls.reference_metadata_file_name)
            top_k_tanimoto_scores.save(store_file_directory / cls.top_k_tanimoto_scores_file_name)
            library_spectrum_set.embeddings.save(store_file_directory / cls.embedding_file_name)
        return cls(ms2deepscore_model, library_spectrum_set.embeddings, top_k_tanimoto_scores, reference_metadata)

    def run_ms2query(
        self,
        query_spectra: Sequence[Spectrum],
        batch_size: int = 1000,
    ) -> pd.DataFrame:

        query_embeddings = Embeddings.create_from_spectra(query_spectra, self.ms2deepscore_model)

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
            score_matrix = cosine_similarity_matrix(selected_query_embeddings, self.reference_embeddings.embeddings)
            highest_score_idx = np.argmax(score_matrix, axis=1)
            library_index_highest_ms2deepscore[start_idx:end_idx] = highest_score_idx

            # get predicted inchikeys
            predicted_inchikeys = self.reference_metadata.iloc[highest_score_idx]["inchikey"]
            # Compute MS2Query reliability score
            ms2query_scores.extend(
                get_ms2query_reliability_prediction(
                    predicted_inchikeys, self.spectrum_indices_per_inchikey, self.top_k_tanimoto_scores, score_matrix
                )
            )

        # construct results df
        results = self.reference_metadata.iloc[library_index_highest_ms2deepscore]
        results["ms2query_reliability_prediction"] = ms2query_scores
        return results


def run_ms2query_from_files(
    query_spectrum_file,
    ms2deepscore_model_file_name,
    reference_embeddings_file,
    top_k_tanimoto_scores_file,
    reference_metadata_file,
    save_file_location,
):
    ms2query_library = MS2QueryLibrary.load_from_files(
        ms2deepscore_model_file_name,
        reference_embeddings_file,
        top_k_tanimoto_scores_file,
        reference_metadata_file,
    )

    query_spectra = list(tqdm(load_spectra(query_spectrum_file), desc="loading_in_query_spectra"))
    results_df = ms2query_library.run_ms2query(query_spectra)
    results_df.to_csv(save_file_location)


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


def extract_metadata_from_library(spectra: AnnotatedSpectrumSet, metadata_to_collect: list):
    collected_metadata = {key: [] for key in metadata_to_collect}
    collected_metadata["spectrum_hashes"] = []
    for spectrum in tqdm(spectra.spectra, desc="Extracting metadata df from spectra"):
        for metadata_key in metadata_to_collect:
            collected_metadata[metadata_key].append(spectrum.get(metadata_key))
        collected_metadata["spectrum_hashes"].append(str(spectrum.__hash__()))
    return pd.DataFrame(collected_metadata)
