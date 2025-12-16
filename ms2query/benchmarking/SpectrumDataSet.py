import copy
from collections import Counter
from typing import Dict, Iterable, List
import numpy as np
from matchms import Spectrum
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi
from ms2deepscore.models import SiameseSpectralModel, compute_embedding_array
from tqdm import tqdm


class SpectrumSetBase:
    """Stores a spectrum dataset making it easy and fast to split on molecules"""

    def __init__(self, spectra: List[Spectrum], progress_bars=False):
        self._spectra = []
        self.spectrum_indexes_per_inchikey = {}
        self.progress_bars = progress_bars
        # init spectra
        self._add_spectra_and_group_per_inchikey(spectra)

    def _add_spectra_and_group_per_inchikey(self, spectra: List[Spectrum]):
        starting_index = len(self._spectra)
        updated_inchikeys = set()
        for i, spectrum in enumerate(
            tqdm(spectra, desc="Adding spectra and grouping per Inchikey", disable=not self.progress_bars)
        ):
            self._spectra.append(spectrum)
            spectrum_index = starting_index + i
            inchikey = spectrum.get("inchikey")[:14]
            updated_inchikeys.add(inchikey)
            if inchikey in self.spectrum_indexes_per_inchikey:
                self.spectrum_indexes_per_inchikey[inchikey].append(spectrum_index)
            else:
                self.spectrum_indexes_per_inchikey[inchikey] = [
                    spectrum_index,
                ]
        return updated_inchikeys

    def add_spectra(self, new_spectra: "SpectrumSetBase"):
        return self._add_spectra_and_group_per_inchikey(new_spectra.spectra)

    def subset_spectra(self, spectrum_indexes) -> "SpectrumSetBase":
        """Returns a new instance of a subset of the spectra"""
        new_instance = copy.copy(self)
        new_instance._spectra = []
        new_instance.spectrum_indexes_per_inchikey = {}
        new_instance._add_spectra_and_group_per_inchikey([self._spectra[index] for index in spectrum_indexes])
        return new_instance

    def spectra_per_inchikey(self, inchikey) -> List[Spectrum]:
        matching_spectra = []
        for index in self.spectrum_indexes_per_inchikey[inchikey]:
            matching_spectra.append(self._spectra[index])
        return matching_spectra

    @property
    def spectra(self):
        return self._spectra

    def copy(self):
        """This copy method ensures all spectra are"""
        new_instance = copy.copy(self)
        new_instance._spectra = self._spectra.copy()
        new_instance.spectrum_indexes_per_inchikey = copy.deepcopy(self.spectrum_indexes_per_inchikey)
        return new_instance


class SpectraWithFingerprints(SpectrumSetBase):
    """Stores a spectrum dataset making it easy and fast to split on molecules"""

    def __init__(self, spectra: List[Spectrum], fingerprint_type="daylight", nbits=4096):
        super().__init__(spectra)
        self.fingerprint_type = fingerprint_type
        self.nbits = nbits
        self.inchikey_fingerprint_pairs: Dict[str, np.array] = {}
        # init spectra
        self.update_fingerprint_per_inchikey(self.spectrum_indexes_per_inchikey.keys())

    def add_spectra(self, new_spectra: "SpectraWithFingerprints"):
        updated_inchikeys = super().add_spectra(new_spectra)
        if hasattr(new_spectra, "inchikey_fingerprint_pairs"):
            if new_spectra.nbits == self.nbits and new_spectra.fingerprint_type == self.fingerprint_type:
                if len(self.inchikey_fingerprint_pairs.keys() & new_spectra.inchikey_fingerprint_pairs.keys()) == 0:
                    self.inchikey_fingerprint_pairs = (
                        self.inchikey_fingerprint_pairs | new_spectra.inchikey_fingerprint_pairs
                    )
                    return
        self.update_fingerprint_per_inchikey(updated_inchikeys)

    def update_fingerprint_per_inchikey(self, inchikeys_to_update: Iterable[str]):
        for inchikey in tqdm(
            inchikeys_to_update, desc="Adding fingerprints to Inchikeys", disable=not self.progress_bars
        ):
            spectra = self.spectra_per_inchikey(inchikey)
            most_common_inchi = Counter([spectrum.get("inchi") for spectrum in spectra]).most_common(1)[0][0]
            fingerprint = _derive_fingerprint_from_inchi(
                most_common_inchi, fingerprint_type=self.fingerprint_type, nbits=self.nbits
            )
            if not isinstance(fingerprint, np.ndarray):
                raise ValueError(f"Fingerprint could not be set for InChI: {most_common_inchi}")
            self.inchikey_fingerprint_pairs[inchikey] = fingerprint

    def copy(self):
        """This copy method ensures all spectra are"""
        new_instance = super().copy()
        new_instance.inchikey_fingerprint_pairs = copy.copy(self.inchikey_fingerprint_pairs)
        return new_instance

    def subset_spectra(self, spectrum_indexes) -> "SpectraWithFingerprints":
        """Returns a new instance of a subset of the spectra"""
        new_instance = super().subset_spectra(spectrum_indexes)
        # Only keep the fingerprints for which we have inchikeys.
        # Important note: This is not a deep copy!
        # And the fingerprint is not reset (so it is not always actually matching the most common inchi)
        new_instance.inchikey_fingerprint_pairs = {inchikey: self.inchikey_fingerprint_pairs[inchikey] for inchikey
                                                   in new_instance.spectrum_indexes_per_inchikey.keys()}
        return new_instance


class SpectraWithMS2DeepScoreEmbeddings(SpectraWithFingerprints):
    def __init__(self, spectra: List[Spectrum], ms2deepscore_model: SiameseSpectralModel, **kwargs):
        super().__init__(spectra, **kwargs)
        self.ms2deepscore_model = ms2deepscore_model
        self.embeddings: np.ndarray = compute_embedding_array(self.ms2deepscore_model, spectra)

    def add_spectra(self, new_spectra: "SpectraWithMS2DeepScoreEmbeddings"):
        super().add_spectra(new_spectra)
        if hasattr(new_spectra, "embeddings"):
            new_embeddings = new_spectra.embeddings
        else:
            new_embeddings = compute_embedding_array(self.ms2deepscore_model, new_spectra.spectra)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def subset_spectra(self, spectrum_indexes) -> "SpectraWithMS2DeepScoreEmbeddings":
        """Returns a new instance of a subset of the spectra"""
        new_instance = super().subset_spectra(spectrum_indexes)
        new_instance.embeddings = self.embeddings[spectrum_indexes]
        return new_instance
