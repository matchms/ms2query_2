from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from matchms.importing import load_spectra
from matchms.Spectrum import Spectrum
from ms2deepscore.models import SiameseSpectralModel, load_model
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2query.ms2query_development.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.ms2query_development.Embeddings import Embeddings, _to_json_serializable
from ms2query.ms2query_development.Fingerprints import Fingerprints
from ms2query.ms2query_development.TopKTanimotoScores import TopKTanimotoScores


class ReferenceLibrary:
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
        "inchi",
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
        self._validate()

    @property
    def reference_metadata(self):
        return self._reference_metadata

    @reference_metadata.setter
    def reference_metadata(self, reference_metadata: pd.DataFrame):
        self._reference_metadata = reference_metadata
        # Get the spectrum_indices_per_inchikey
        self.spectrum_indices_per_inchikey = defaultdict(list)
        for lib_spec_index, inchikey in enumerate(self.reference_metadata["inchikey"]):
            self.spectrum_indices_per_inchikey[inchikey[:14]].append(lib_spec_index)

    def _validate(self):
        # Check that the loaded files match
        if (
            _to_json_serializable(self.ms2deepscore_model.model_settings.get_dict())
            != self.reference_embeddings.model_settings
        ):
            raise ValueError(
                "The settings of the ms2deepscore model do not match the model used for creating the library embeddings"
            )
        if list(self.reference_metadata["spectrum_hashes"]) != [
            str(spectrum_hash) for spectrum_hash in self.reference_embeddings.index_to_spectrum_hash
        ]:
            raise ValueError("The loaded metadata does not match the used embeddings")
        if set(self.spectrum_indices_per_inchikey.keys()) != set(
            self.top_k_tanimoto_scores.top_k_inchikeys_and_scores.index
        ):
            raise ValueError("The inchikeys in the metadata and in the top_k_tanimoto_scores do not match")

    @classmethod
    def load_from_directory(cls, library_file_directory) -> "ReferenceLibrary":
        library_file_directory = Path(library_file_directory)
        reference_embeddings_file = library_file_directory / cls.embedding_file_name
        top_k_tanimoto_scores_file = library_file_directory / cls.top_k_tanimoto_scores_file_name
        reference_metadata_file = library_file_directory / cls.reference_metadata_file_name
        ms2deepscore_model_file_name = library_file_directory / cls.ms2deepscore_model_file_name
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
    ) -> "ReferenceLibrary":
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
    ) -> "ReferenceLibrary":
        """Creates all the files needed for MS2Query and stores them"""
        # library_spectra = list(tqdm(load_spectra(library_spectra_file), "Loading library spectra"))
        library_spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(library_spectra)
        ms2deepscore_model = load_model(ms2deepscore_model_file_name)
        reference_metadata = extract_metadata_from_library(
            library_spectrum_set,
            cls.metadata_to_store,
        )
        fingerprints = Fingerprints.from_dataframe(reference_metadata, cls.fingerprint_type, cls.fingerprint_nbits)
        top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
            fingerprints, fingerprints, cls.top_k_inchikeys
        )
        library_spectrum_set.add_embeddings(ms2deepscore_model)
        return cls(ms2deepscore_model, library_spectrum_set.embeddings, top_k_tanimoto_scores, reference_metadata)

    def add_spectra(self, new_library_spectra):
        """Add spectra to the already existing library (the ms2deepscore model won't be retrained)"""
        # Check that no duplicates are added
        hashes = [spectrum.__hash__() for spectrum in tqdm(new_library_spectra, desc="Hashing spectra")]
        if len(hashes) != len(set(hashes)):
            raise ValueError("There are duplicated spectra, please make sure there are no duplicates")
        # Only add spectra not already in the library
        existing_hashes = set(self.reference_metadata["spectrum_hashes"])
        new_spectra = [spectrum for spectrum, h in zip(new_library_spectra, hashes) if h not in existing_hashes]
        if len(new_spectra) != len(new_library_spectra):
            print(f"{len(new_library_spectra) - len(new_spectra)} were not added, since already in the library")

        new_spectrum_set = AnnotatedSpectrumSet.create_spectrum_set(new_spectra)

        # Add spectrum metadata
        reference_metadata = extract_metadata_from_library(
            new_spectrum_set,
            self.metadata_to_store,
        )
        self.reference_metadata = pd.concat([self.reference_metadata, reference_metadata], ignore_index=True)

        # Add embeddings
        new_spectrum_set.add_embeddings(self.ms2deepscore_model)
        self.reference_embeddings = self.reference_embeddings + new_spectrum_set.embeddings

        # Recompute top k tanimoto scores
        fingerprints = Fingerprints.from_dataframe(
            self.reference_metadata, self.fingerprint_type, self.fingerprint_nbits
        )
        self.top_k_tanimoto_scores = TopKTanimotoScores.calculate_from_fingerprints(
            fingerprints, fingerprints, self.top_k_inchikeys
        )

    def save(self, store_file_directory: str | Path):
        store_file_directory = Path(store_file_directory)
        store_file_directory.mkdir(parents=True, exist_ok=True)

        def file_does_not_exist_yet(file_name: Path):
            if file_name.exists():
                print(f"The file: {file_name} already exists, not saved.")
                return False
            return True

        # Save files after checking it does not exist yet
        if file_does_not_exist_yet(store_file_directory / self.reference_metadata_file_name):
            self.reference_metadata.to_parquet(store_file_directory / self.reference_metadata_file_name)

        if file_does_not_exist_yet(store_file_directory / self.top_k_tanimoto_scores_file_name):
            self.top_k_tanimoto_scores.save(store_file_directory / self.top_k_tanimoto_scores_file_name)

        if file_does_not_exist_yet(store_file_directory / self.embedding_file_name):
            self.reference_embeddings.save(store_file_directory / self.embedding_file_name)

        if file_does_not_exist_yet(store_file_directory / self.ms2deepscore_model_file_name):
            self.ms2deepscore_model.save(store_file_directory / self.ms2deepscore_model_file_name)

    def run_ms2query(
        self,
        query_spectra: Sequence[Spectrum],
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        return self._run_ms2query(
            query_spectra, self.reference_embeddings.embeddings, self.reference_metadata, batch_size
        )

    def _run_ms2query(
        self, query_spectra, reference_embeddings: np.ndarray, reference_metadata: pd.DataFrame, batch_size
    ):
        query_embeddings = Embeddings.create_from_spectra(query_spectra, self.ms2deepscore_model)

        library_index_highest_ms2deepscore, highest_ms2deepscore = get_highest_ms2deepscore(
            query_embeddings.embeddings, reference_embeddings, batch_size
        )

        # construct results df
        results = reference_metadata.iloc[library_index_highest_ms2deepscore].copy()
        # Compute MS2Query reliability score
        results["ms2query_reliability_prediction"] = self.get_ms2query_reliability_prediction(
            results["inchikey"],
            query_embeddings.embeddings,
        )
        results["highest_ms2deepscore"] = highest_ms2deepscore
        # Add spectrum metadata
        results["query_precursor_mz"] = [spectrum.get("precursor_mz") for spectrum in query_spectra]
        results["query_retention_time"] = [spectrum.get("retention_time") for spectrum in query_spectra]
        return results

    def run_semi_targeted_ms2query(
        self,
        query_spectra: Sequence[Spectrum],
        inchikeys_to_check: set[str],
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """Does a search in just a subset of the inchikeys in the library.

        This is a semi targeted analogue search, where you specifically search for a subset of inchikeys.
        The rest of the library is still used to compute the MS2Query reliability score."""
        inchikey_14_to_check = {inchikey[:14] for inchikey in inchikeys_to_check}

        spectrum_indices_inchikeys_to_check = []
        for inchikey in inchikey_14_to_check:
            if inchikey in self.spectrum_indices_per_inchikey:
                spectrum_indices_inchikeys_to_check.extend(self.spectrum_indices_per_inchikey[inchikey])
            else:
                print(f"The inchikey: {inchikey} is not in the reference library, so it won't be searched for")
        embeddings_inchikeys_to_check = self.reference_embeddings.subset_embeddings_from_index(
            spectrum_indices_inchikeys_to_check
        )
        metadata_subset = self.reference_metadata.iloc[spectrum_indices_inchikeys_to_check]
        return self._run_ms2query(query_spectra, embeddings_inchikeys_to_check.embeddings, metadata_subset, batch_size)

    def get_ms2query_reliability_prediction(
        self,
        predicted_inchikeys: Iterable[str],
        query_embeddings,
    ) -> list[float]:
        ms2query_scores = []
        for query_spectrum_index, library_inchikey in enumerate(predicted_inchikeys):
            top_k_inchikeys = self.top_k_tanimoto_scores.select_top_k_inchikeys(library_inchikey[:14])
            maximum_ms2deepscores = np.zeros(self.top_k_tanimoto_scores.k, dtype=float)
            for i, inchikey in enumerate(top_k_inchikeys):
                spectrum_indexes = self.spectrum_indices_per_inchikey[inchikey]
                score_matrix = cosine_similarity_matrix(
                    query_embeddings[[query_spectrum_index], :], self.reference_embeddings.embeddings[spectrum_indexes]
                )
                highest_ms2deepscore = np.max(score_matrix)
                maximum_ms2deepscores[i] = highest_ms2deepscore
            ms2query_scores.append(np.mean(maximum_ms2deepscores))
        return ms2query_scores


def get_highest_ms2deepscore(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, batch_size):
    num_of_query_embeddings = query_embeddings.shape[0]

    library_index_highest_ms2deepscore = np.zeros((num_of_query_embeddings), dtype=int)
    highest_ms2deepscore = np.zeros((num_of_query_embeddings), dtype=float)
    for start_idx in tqdm(
        range(0, num_of_query_embeddings, batch_size),
        desc="Predicting highest ms2deepscore per batch of "
        + str(min(batch_size, num_of_query_embeddings))
        + " embeddings",
    ):
        # Do MS2DeepScore predictions for batch
        end_idx = min(start_idx + batch_size, num_of_query_embeddings)
        selected_query_embeddings = query_embeddings[start_idx:end_idx]
        score_matrix = cosine_similarity_matrix(selected_query_embeddings, reference_embeddings)
        highest_score_idx = np.argmax(score_matrix, axis=1)
        library_index_highest_ms2deepscore[start_idx:end_idx] = highest_score_idx
        highest_ms2deepscore[start_idx:end_idx] = np.max(score_matrix, axis=1)
    return library_index_highest_ms2deepscore, highest_ms2deepscore


def run_ms2query_from_files(
    query_spectrum_file,
    ms2deepscore_model_file_name,
    reference_embeddings_file,
    top_k_tanimoto_scores_file,
    reference_metadata_file,
    save_file_location,
):
    ms2query_library = ReferenceLibrary.load_from_files(
        ms2deepscore_model_file_name,
        reference_embeddings_file,
        top_k_tanimoto_scores_file,
        reference_metadata_file,
    )

    query_spectra = list(tqdm(load_spectra(query_spectrum_file), desc="loading_in_query_spectra"))
    results_df = ms2query_library.run_ms2query(query_spectra)
    results_df.to_csv(save_file_location)


def extract_metadata_from_library(spectra: AnnotatedSpectrumSet, metadata_to_collect: list):
    collected_metadata = {key: [] for key in metadata_to_collect}
    collected_metadata["spectrum_hashes"] = []
    for spectrum in tqdm(spectra.spectra, desc="Extracting metadata df from spectra"):
        for metadata_key in metadata_to_collect:
            collected_metadata[metadata_key].append(spectrum.get(metadata_key))
        collected_metadata["spectrum_hashes"].append(str(spectrum.__hash__()))
    return pd.DataFrame(collected_metadata)
