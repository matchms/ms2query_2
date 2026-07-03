import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel, compute_embedding_array
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm


class Embeddings:
    """Stores Embeddings for a list of mass spectra"""

    def __init__(self, embeddings: np.ndarray, spectrum_hashes: tuple, model_settings: dict):
        if len(spectrum_hashes) != embeddings.shape[0]:
            raise ValueError("Number of spectra hashes does not match number of embeddings")
        self.index_to_spectrum_hash = spectrum_hashes
        self._spectrum_hash_to_index = {
            spectrum_hash: index for index, spectrum_hash in enumerate(self.index_to_spectrum_hash)
        }
        self.model_settings = model_settings
        self._embeddings = embeddings

    @classmethod
    def create_from_spectra(cls, spectra: Sequence[Spectrum], model: SiameseSpectralModel) -> "Embeddings":
        index_to_spectrum_hash = tuple(spectrum.__hash__() for spectrum in tqdm(spectra, desc="Hashing spectra"))
        if len(set(index_to_spectrum_hash)) != len(spectra):
            raise ValueError("There are duplicated spectra in the spectrum list")

        model_settings = model.model_settings.get_dict()
        embeddings: np.ndarray = compute_embedding_array(model, spectra)  # type: ignore
        return cls(embeddings, index_to_spectrum_hash, model_settings)

    def __add__(self, other: "Embeddings") -> "Embeddings":
        if not isinstance(other, Embeddings):
            return NotImplemented
        if self.model_settings != other.model_settings:
            raise ValueError("Model settings of merged embeddings do not match")
        if not set(self.index_to_spectrum_hash).isdisjoint(other.index_to_spectrum_hash):
            raise ValueError("There are repeated spectra in the embeddings that are added together")
        combined_embeddings = np.vstack([self._embeddings, other._embeddings])
        index_to_spectrum_hash = self.index_to_spectrum_hash + other.index_to_spectrum_hash
        return Embeddings(combined_embeddings, index_to_spectrum_hash, self.model_settings)

    def subset_embeddings(self, spectra):
        spectrum_hashes = tuple(spectrum.__hash__() for spectrum in spectra)
        try:
            embedding_indexes = [self._spectrum_hash_to_index[spectrum_hash] for spectrum_hash in spectrum_hashes]
        except KeyError:
            raise ValueError("The given spectra are not stored in Embeddings")
        embeddings = self._embeddings[embedding_indexes].copy()
        return Embeddings(embeddings, spectrum_hashes, self.model_settings)

    @property
    def embeddings(self):
        return self._embeddings.view()

    @property
    def model_settings(self):
        return self._model_settings.copy()

    @model_settings.setter
    def model_settings(self, model_settings):
        self._model_settings: dict = _to_json_serializable(model_settings)

    def copy(self) -> "Embeddings":
        return Embeddings(
            embeddings=self._embeddings.copy(),
            spectrum_hashes=tuple(self.index_to_spectrum_hash),
            model_settings=dict(self.model_settings),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Embeddings):
            return NotImplemented
        if self.model_settings != other.model_settings:
            print("Model setting not equal")
            return False
        if self.index_to_spectrum_hash != other.index_to_spectrum_hash:
            print("index to spectrum hash not equal")
            return False
        return np.array_equal(self.embeddings, other.embeddings)

    def save(self, path: str | Path) -> None:
        """Save embeddings to a .npz file with metadata stored alongside.

        Args:
            path: File path. A '.npz' extension will be added if not present.
        """
        path = Path(path).with_suffix(".npz")
        metadata = {
            "model_settings": self.model_settings,
            "index_to_spectrum_hash": list(self.index_to_spectrum_hash),
        }
        np.savez_compressed(
            path,
            embeddings=self._embeddings,
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Embeddings":
        """Load embeddings from a saved .npz file.

        Args:
            path: Path to the saved .npz file.
        """
        path = Path(path).with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            metadata = json.loads(data["metadata"].item())
        return cls(
            embeddings=embeddings,
            spectrum_hashes=tuple(metadata["index_to_spectrum_hash"]),
            model_settings=metadata["model_settings"],
        )


def calculate_ms2deepscore_df(query_embeddings: Embeddings, library_embeddings: Embeddings):
    """Returns a DF, where the indexes and column labels are the spectrum hashes"""
    ms2deepscores = cosine_similarity_matrix(query_embeddings.embeddings, library_embeddings.embeddings)
    return pd.DataFrame(
        ms2deepscores, index=query_embeddings.index_to_spectrum_hash, columns=library_embeddings.index_to_spectrum_hash
    )


def _to_json_serializable(obj):
    """Changes a dict to be json sericalizable, so it is the same when loaded"""
    if isinstance(obj, dict):
        return {key: _to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
