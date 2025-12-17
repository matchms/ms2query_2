import copy
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional
import numpy as np
from matchms import Spectrum
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi
from ms2deepscore.models import SiameseSpectralModel, compute_embedding_array
from tqdm import tqdm


class SpectrumSet:
    """Stores a spectrum dataset making it easy and fast to split on molecules"""

    def __init__(self, spectra: List[Spectrum], progress_bars=False):
        self._spectra = []
        self.spectrum_indexes_per_inchikey = defaultdict(list)
        self.progress_bars = progress_bars
        # init spectra
        self._add_spectra_and_group_per_inchikey(spectra)
        self.most_common_inchi_per_inchikey = {}
        self._update_most_common_inchi_per_inchikey(list(self.spectrum_indexes_per_inchikey.keys()))

    def add_spectra(self, new_spectra: "SpectrumSet"):
        updated_inchikeys = self._add_spectra_and_group_per_inchikey(new_spectra.spectra)
        self._update_most_common_inchi_per_inchikey(updated_inchikeys)

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
            self.spectrum_indexes_per_inchikey[inchikey].append(spectrum_index)
        return updated_inchikeys

    def _update_most_common_inchi_per_inchikey(self, new_inchikeys):
        for inchikey in tqdm(new_inchikeys, desc="Get most common inchi per inchikey"):
            spectra_matching_inchikey = self.spectra_per_inchikey(inchikey)
            most_common_inchi = Counter([spectrum.get("inchi") for spectrum in spectra_matching_inchikey]).most_common(1)[0][0]
            self.most_common_inchi_per_inchikey[inchikey](most_common_inchi)

    def subset_spectra(self, spectrum_indexes) -> "SpectrumSet":
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

class Fingerprints:
    def __init__(self, most_common_inchi_per_inchikey, fingerprint_type, nbits):
        self.most_common_inchi_per_inchikey = most_common_inchi_per_inchikey
        self.index_to_inchikey = list(self.most_common_inchi_per_inchikey.keys())
        self.inchikey_to_index = {inchikey: index for index, inchikey in enumerate(self.index_to_inchikey)}
        self.fingerprint_type = fingerprint_type
        self.nbits = nbits

        self.fingerprints = np.zeros((len(self.index_to_inchikey), self.nbits), dtype=np.uint8)
        self.update_fingerprints(self.index_to_inchikey)

    def update_fingerprints(self, inchikeys: Iterable[str]):
        for inchikey in tqdm(inchikeys, desc="Adding fingerprints to Inchikeys"):
            inchikey_index = self.inchikey_to_index[inchikey]
            self.fingerprints[inchikey_index, :] = self.compute_fingerprint(inchikey)

    def compute_fingerprint(self, inchikey):
        most_common_inchi = self.most_common_inchi_per_inchikey[inchikey]
        fingerprint = _derive_fingerprint_from_inchi(
            most_common_inchi, fingerprint_type=self.fingerprint_type, nbits=self.nbits)
        if not isinstance(fingerprint, np.ndarray):
            raise ValueError(f"Fingerprint could not be set for InChI: {most_common_inchi}")
        return fingerprint

    def get_fingerprints(self, list_of_inchikeys):
        list_of_indexes = [self.inchikey_to_index[inchikey] for inchikey in list_of_inchikeys]
        return self.fingerprints[list_of_indexes]

    def add_new_inchikeys(self, new_most_common_inchi_per_inchikey: dict[str, str]):
        inchikeys_to_update = []
        inchikeys_to_add = []
        for inchikey, inchi in new_most_common_inchi_per_inchikey.items():
            if inchikey in self.most_common_inchi_per_inchikey:
                if self.most_common_inchi_per_inchikey[inchikey] == inchi:
                    # the inchikey is unchanged
                    continue
            else:
                inchikeys_to_add.append(inchikey)
            self.most_common_inchi_per_inchikey[inchikey] = inchi
            inchikeys_to_update.append(inchikey)

        # Add the inchikeys_to_add
        if inchikeys_to_add:
            self.index_to_inchikey.extend(inchikeys_to_add)
            self.inchikey_to_index = {inchikey: index for index, inchikey in enumerate(self.index_to_inchikey)}
            new_rows = np.zeros((len(inchikeys_to_add), self.nbits), dtype=np.uint8)
            self.fingerprints = np.vstack([self.fingerprints, new_rows])

        self.update_fingerprints(inchikeys_to_update)

class SpectraWithFingerprints(SpectrumSet):
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
