from collections import defaultdict
from typing import Iterable, List, Mapping, Optional, Sequence
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel
from tqdm import tqdm
from ms2query.benchmarking.Embeddings import Embeddings


class AnnotatedSpectrumSet:
    """Stores a spectrum dataset making it easy and fast to split on molecules"""

    def __init__(
        self,
        spectra: Sequence[Spectrum],
        spectrum_indices_per_inchikey: Mapping[str, Iterable[int]],
        embeddings: Optional[Embeddings] = None,
    ):
        self._spectra = tuple([spectrum.clone() for spectrum in spectra])
        self.spectrum_indices_per_inchikey: dict[str, tuple[int, ...]] = {
            key: tuple(values) for key, values in spectrum_indices_per_inchikey.items()
        }
        self._embeddings = embeddings

    @classmethod
    def create_spectrum_set(cls, spectra: Sequence[Spectrum]) -> "AnnotatedSpectrumSet":
        spectrum_indices_per_inchikey = defaultdict(list)
        for spectrum_index, spectrum in enumerate(tqdm(spectra, desc="Create mapping from inchikey to spectrum")):
            inchikey = spectrum.get("inchikey")
            if inchikey is None:
                raise ValueError("Annotated Spectrum set expects spectra that all have an inchikey")
            spectrum_indices_per_inchikey[inchikey[:14]].append(spectrum_index)
        return cls(spectra, spectrum_indices_per_inchikey)

    def __add__(self, other) -> "AnnotatedSpectrumSet":
        """Adds two spectrum sets together"""
        if not isinstance(other, AnnotatedSpectrumSet):
            return NotImplemented
        spectra = self.spectra + other.spectra
        # update spectrum_indices_per_inchikey
        starting_index = len(self.spectra)
        reindexed_indices_per_inchikey = {}
        for inchikey, list_of_spectrum_indices in other.spectrum_indices_per_inchikey.items():
            reindexed_indices_per_inchikey[inchikey] = [v + starting_index for v in list_of_spectrum_indices]
        # combine indices
        spectrum_indices_per_inchikey = defaultdict(list)
        for indices_per_inchikey in (self.spectrum_indices_per_inchikey, reindexed_indices_per_inchikey):
            for inchikey, indices in indices_per_inchikey.items():
                spectrum_indices_per_inchikey[inchikey].extend(indices)

        # combine embeddings
        embeddings = None
        if self._embeddings and other._embeddings:
            embeddings = Embeddings.combine_embeddings(self.embeddings, other.embeddings)
        return AnnotatedSpectrumSet(spectra, spectrum_indices_per_inchikey, embeddings=embeddings)

    def subset_spectra(self, spectrum_indices) -> "AnnotatedSpectrumSet":
        """Returns a new instance of a subset of the spectra"""
        spectra = [self._spectra[index] for index in spectrum_indices]
        new_instance = AnnotatedSpectrumSet.create_spectrum_set(spectra)
        if self._embeddings is not None:
            new_instance._embeddings = self.embeddings.subset_embeddings(spectra)
        return new_instance

    def spectra_per_inchikey(self, inchikey) -> List[Spectrum]:
        matching_spectra = []
        for index in self.spectrum_indices_per_inchikey[inchikey]:
            matching_spectra.append(self._spectra[index])
        return matching_spectra

    def add_embeddings(self, model: SiameseSpectralModel):
        self._embeddings = Embeddings.create_from_spectra(self._spectra, model)

    @property
    def spectra(self):
        return self._spectra

    @property
    def embeddings(self) -> "Embeddings":
        if self._embeddings is None:
            raise ValueError("First run the 'add_embeddings' method")
        return self._embeddings

    @property
    def inchikeys(self):
        return tuple(self.spectrum_indices_per_inchikey.keys())

    def __copy__(self):
        return AnnotatedSpectrumSet(self.spectra, self.spectrum_indices_per_inchikey, self.embeddings)

    def __eq__(self, other: object):
        if not isinstance(other, AnnotatedSpectrumSet):
            raise ValueError("__Eq__ can only be done between two AnnotatedSpectrumSets")
        if self.spectra != other.spectra:
            return False
        if self.spectrum_indices_per_inchikey != other.spectrum_indices_per_inchikey:
            return False
        if self._embeddings != other._embeddings:
            return False
        return True

    def __len__(self):
        return len(self._spectra)
