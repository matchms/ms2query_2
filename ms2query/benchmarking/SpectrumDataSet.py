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

        self._fingerprints = None
        self._embeddings = None

    def add_spectra(self, new_spectra: "SpectrumSet"):
        updated_inchikeys = self._add_spectra_and_group_per_inchikey(new_spectra.spectra)
        self._update_most_common_inchi_per_inchikey(updated_inchikeys)
        if self.embeddings is not None:
            self.embeddings.add_embeddings(new_spectra.embeddings)
        if self.fingerprints is not None:
            self.fingerprints.add_new_inchikeys(new_spectra.most_common_inchi_per_inchikey)

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
        spectra = [self._spectra[index] for index in spectrum_indexes]
        new_instance = SpectrumSet(spectra, progress_bars=self.progress_bars)
        if self.embeddings is not None:
            new_instance._embeddings = self.embeddings.get_embeddings(spectra)
        if self.fingerprints is not None:
            inchikeys = [spectrum.get("inchikey")[:14] for spectrum in spectra]
            new_instance._fingerprints = self.fingerprints.subset_fingerprints(inchikeys)
        return new_instance

    def spectra_per_inchikey(self, inchikey) -> List[Spectrum]:
        matching_spectra = []
        for index in self.spectrum_indexes_per_inchikey[inchikey]:
            matching_spectra.append(self._spectra[index])
        return matching_spectra

    def add_embeddings(self, model: SiameseSpectralModel):
        self._embeddings = Embeddings(self._spectra, model)

    def add_fingerprints(self, fingerprint_type, nbits):
        self._fingerprints = Fingerprints(self.most_common_inchi_per_inchikey, fingerprint_type, nbits)

    @property
    def spectra(self):
        return self._spectra

    @property
    def fingerprints(self):
        return self._fingerprints

    @property
    def embeddings(self):
        return self._embeddings

    def copy(self):
        """This copy method ensures all spectra are"""
        new_instance = copy.copy(self)
        new_instance._spectra = self._spectra.copy()
        new_instance.spectrum_indexes_per_inchikey = copy.deepcopy(self.spectrum_indexes_per_inchikey)
        return new_instance

class Fingerprints:
    # I just realize that there already exists a Fingerprints class in matchms, with almost the same functionality,
    # so it will be good to merge both. The matchms version works slighlty different.
    def __init__(self, most_common_inchi_per_inchikey, fingerprint_type, nbits):
        self.most_common_inchi_per_inchikey = most_common_inchi_per_inchikey
        self.fingerprint_type = fingerprint_type
        self.nbits = nbits

        self._index_to_inchikey = []
        self._inchikey_to_index = dict()

        self._add_inchikeys_to_index(list(self.most_common_inchi_per_inchikey.keys()))

        self.fingerprints = np.zeros((len(self.index_to_inchikey), self.nbits), dtype=np.uint8)
        self.update_fingerprints(self.index_to_inchikey)

    def update_fingerprints(self, inchikeys: Iterable[str]):
        for inchikey in tqdm(inchikeys, desc="Adding fingerprints to Inchikeys"):
            inchikey_index = self._inchikey_to_index[inchikey]
            self.fingerprints[inchikey_index, :] = self.compute_fingerprint(inchikey)

    def compute_fingerprint(self, inchikey):
        most_common_inchi = self.most_common_inchi_per_inchikey[inchikey]
        fingerprint = _derive_fingerprint_from_inchi(
            most_common_inchi, fingerprint_type=self.fingerprint_type, nbits=self.nbits)
        if not isinstance(fingerprint, np.ndarray):
            raise ValueError(f"Fingerprint could not be set for InChI: {most_common_inchi}")
        return fingerprint

    def get_fingerprints(self, list_of_inchikeys):
        list_of_indexes = [self._inchikey_to_index[inchikey] for inchikey in list_of_inchikeys]
        return self.fingerprints[list_of_indexes]

    def subset_fingerprints(self, list_of_inchikeys) -> "Fingerprints":
        new_instance = Fingerprints(dict(), self.fingerprint_type, self.nbits)
        new_instance.fingerprints = self.get_fingerprints(list_of_inchikeys)

        new_most_common_inchi_per_inchikey = dict()
        for inchikey in list_of_inchikeys:
            new_most_common_inchi_per_inchikey[inchikey] = self.most_common_inchi_per_inchikey[inchikey]
        new_instance.most_common_inchi_per_inchikey = new_most_common_inchi_per_inchikey
        new_instance._add_inchikeys_to_index(list_of_inchikeys)
        return new_instance

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
            self._add_inchikeys_to_index(inchikeys_to_add)
            new_rows = np.zeros((len(inchikeys_to_add), self.nbits), dtype=np.uint8)
            self.fingerprints = np.vstack([self.fingerprints, new_rows])

        self.update_fingerprints(inchikeys_to_update)

    @property
    def index_to_inchikey(self):
        return self._index_to_inchikey

    def _add_inchikeys_to_index(self, inchikeys_to_add):
        self._index_to_inchikey.extend(inchikeys_to_add)
        self._inchikey_to_index = {inchikey: index for index, inchikey in enumerate(self.index_to_inchikey)}


class Embeddings:
    """Stores Embeddings for a list of mass spectra"""
    def __init__(self, spectra: List[Spectrum],
                 model: SiameseSpectralModel):
        self.index_to_spectrum_hash = [spectrum.__hash__() for spectrum in spectra]
        if set(self.index_to_spectrum_hash) != len(spectra):
            raise ValueError("There are duplicated spectra in the spectrum list")
        self.spectrum_hash_to_index = {spectrum_hash: index for index, spectrum_hash in enumerate(self.index_to_spectrum_hash)}

        self.model_settings = model.model_settings
        self.embeddings: np.ndarray = compute_embedding_array(model, spectra)

    def add_embeddings(self, embeddings: "Embeddings"):
        if embeddings.model_settings != self.model_settings:
            raise ValueError("Model settings of merged embeddings do not match")
        if not set(embeddings.spectrum_hash_to_index).isdisjoint(self.spectrum_hash_to_index):
            raise ValueError("There are repeated spectra in the embeddings that are added together")
        self.embeddings = np.vstack([self.embeddings, embeddings])
        self.spectrum_hash_to_index += embeddings.spectrum_hash_to_index
        self.spectrum_hash_to_index = {spectrum_hash: index for index, spectrum_hash in enumerate(self.index_to_spectrum_hash)}

    def get_embeddings(self, spectra: list[Spectrum]):
        embedding_indexes = []
        for spectrum in spectra:
            embedding_indexes.append(self.spectrum_hash_to_index[spectrum.__hash__()])
        return self.embeddings[embedding_indexes]
