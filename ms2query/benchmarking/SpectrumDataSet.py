import copy
from collections import Counter, defaultdict
from typing import List
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel
from tqdm import tqdm

from ms2query.benchmarking.Embeddings import Embeddings
from ms2query.benchmarking.Fingerprints import Fingerprints


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
        if self._embeddings is not None:
            self._embeddings = Embeddings.combine_embeddings(self.embeddings, new_spectra.embeddings)
        if self._fingerprints is not None:
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
            self.most_common_inchi_per_inchikey[inchikey] = most_common_inchi

    def subset_spectra(self, spectrum_indexes) -> "SpectrumSet":
        """Returns a new instance of a subset of the spectra"""
        spectra = [self._spectra[index] for index in spectrum_indexes]
        new_instance = SpectrumSet(spectra, progress_bars=self.progress_bars)
        if self._embeddings is not None:
            new_instance._embeddings = self.embeddings.subset_embeddings(spectra)
        if self._fingerprints is not None:
            inchikeys = [spectrum.get("inchikey")[:14] for spectrum in spectra]
            new_instance._fingerprints = self.fingerprints.subset_fingerprints(inchikeys)
        return new_instance

    def spectra_per_inchikey(self, inchikey) -> List[Spectrum]:
        matching_spectra = []
        for index in self.spectrum_indexes_per_inchikey[inchikey]:
            matching_spectra.append(self._spectra[index])
        return matching_spectra

    def add_embeddings(self, model: SiameseSpectralModel):
        self._embeddings = Embeddings.create_from_spectra(self._spectra, model)

    def add_fingerprints(self, fingerprint_type, nbits):
        self._fingerprints = Fingerprints(self.most_common_inchi_per_inchikey, fingerprint_type, nbits)

    @property
    def spectra(self):
        return self._spectra

    @property
    def fingerprints(self) -> "Fingerprints":
        if self._fingerprints is None:
            raise ValueError("First run add_fingerprints")
        return self._fingerprints

    @property
    def embeddings(self) -> "Embeddings":
        if self._embeddings is None:
            raise ValueError("First run add_embeddings")
        return self._embeddings

    def copy(self):
        """This copy method ensures all spectra are"""
        new_instance = copy.copy(self)
        new_instance._spectra = self._spectra.copy()
        new_instance.spectrum_indexes_per_inchikey = copy.deepcopy(self.spectrum_indexes_per_inchikey)
        return new_instance


