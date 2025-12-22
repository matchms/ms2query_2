from typing import List

import numpy as np
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel, compute_embedding_array


class Embeddings:
    """Stores Embeddings for a list of mass spectra"""
    def __init__(self, embeddings: np.ndarray, spectrum_hashes: list[int], model_settings: dict):
        if len(spectrum_hashes) != embeddings.shape[0]:
            raise ValueError("Number of spectra hashes does not match number of embeddings")
        self._index_to_spectrum_hash = spectrum_hashes
        self._spectrum_hash_to_index = {spectrum_hash: index for index, spectrum_hash in enumerate(self._index_to_spectrum_hash)}
        self._model_settings = model_settings
        self._embeddings = embeddings

    @classmethod
    def create_from_spectra(cls, spectra: List[Spectrum],
                 model: SiameseSpectralModel):
        index_to_spectrum_hash = [spectrum.__hash__() for spectrum in spectra]
        if len(set(index_to_spectrum_hash)) != len(spectra):
            raise ValueError("There are duplicated spectra in the spectrum list")

        model_settings = model.model_settings.get_dict()
        embeddings = compute_embedding_array(model, spectra)
        return cls(embeddings, index_to_spectrum_hash, model_settings)


    def add_embeddings(self, embeddings: "Embeddings"):
        if embeddings._model_settings != self.model_settings:
            raise ValueError("Model settings of merged embeddings do not match")
        if not set(embeddings._spectrum_hash_to_index).isdisjoint(self._spectrum_hash_to_index):
            raise ValueError("There are repeated spectra in the embeddings that are added together")
        self._embeddings = np.vstack([self._embeddings, embeddings.embeddings])
        self._index_to_spectrum_hash += embeddings._index_to_spectrum_hash
        self._spectrum_hash_to_index = {spectrum_hash: index for index, spectrum_hash in enumerate(self._index_to_spectrum_hash)}

    def get_embeddings(self, spectra) -> np.ndarray:
        embedding_indexes = []
        for spectrum in spectra:
            embedding_indexes.append(self._spectrum_hash_to_index[spectrum.__hash__()])
        embeddings = self._embeddings[embedding_indexes]
        return embeddings

    def subset_embeddings(self, spectra):
        spectrum_hashes = [spectrum.__hash__() for spectrum in spectra]
        embedding_indexes = [self._spectrum_hash_to_index[spectrum_hash] for spectrum_hash in spectrum_hashes]
        embeddings = self._embeddings[embedding_indexes].copy()
        return Embeddings(embeddings, spectrum_hashes, self.model_settings)

    @property
    def embeddings(self):
        return self._embeddings.view()

    @property
    def model_settings(self):
        return self._model_settings.copy()

    def copy(self) -> "Embeddings":
        return Embeddings(
            embeddings=self._embeddings.copy(),
            spectrum_hashes=list(self._index_to_spectrum_hash),
            model_settings=dict(self._model_settings),
        )
