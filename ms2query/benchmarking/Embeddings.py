from typing import List

import numpy as np
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel, compute_embedding_array


class Embeddings:
    """Stores Embeddings for a list of mass spectra"""
    def __init__(self, embeddings: np.ndarray, spectrum_hashes: tuple, model_settings: dict):
        if len(spectrum_hashes) != embeddings.shape[0]:
            raise ValueError("Number of spectra hashes does not match number of embeddings")
        self.index_to_spectrum_hash = spectrum_hashes
        self._spectrum_hash_to_index = {spectrum_hash: index for index, spectrum_hash in enumerate(self.index_to_spectrum_hash)}
        self._model_settings = model_settings
        self._embeddings = embeddings

    @classmethod
    def create_from_spectra(cls, spectra: List[Spectrum],
                 model: SiameseSpectralModel) -> "Embeddings":
        index_to_spectrum_hash = tuple(spectrum.__hash__() for spectrum in spectra)
        if len(set(index_to_spectrum_hash)) != len(spectra):
            raise ValueError("There are duplicated spectra in the spectrum list")

        model_settings = model.model_settings.get_dict()
        embeddings = compute_embedding_array(model, spectra)
        return cls(embeddings, index_to_spectrum_hash, model_settings)

    @classmethod
    def combine_embeddings(cls, embeddings_1, embeddings_2) -> "Embeddings":
        if embeddings_1.model_settings != embeddings_2.model_settings:
            raise ValueError("Model settings of merged embeddings do not match")
        if not set(embeddings_1.spectrum_hash_to_index).isdisjoint(embeddings_2.spectrum_hash_to_index):
            raise ValueError("There are repeated spectra in the embeddings that are added together")
        combined_embeddings =  np.vstack([embeddings_1.embeddings, embeddings_2.embeddings])
        index_to_spectrum_hash = embeddings_1.index_to_spectrum_hash + embeddings_2.index_to_spectrum_hash
        return cls(combined_embeddings, index_to_spectrum_hash, embeddings_1.model_settings)

    def subset_embeddings(self, spectra):
        spectrum_hashes = tuple(spectrum.__hash__() for spectrum in spectra)
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
            spectrum_hashes=tuple(self.index_to_spectrum_hash),
            model_settings=dict(self._model_settings),
        )
