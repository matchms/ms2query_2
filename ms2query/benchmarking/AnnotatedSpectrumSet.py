from collections import defaultdict
from typing import List, Iterable, Optional, Sequence
from matchms import Spectrum
from ms2deepscore.models import SiameseSpectralModel

from ms2query.benchmarking.Embeddings import Embeddings


class AnnotatedSpectrumSet:
    """Stores a spectrum dataset making it easy and fast to split on molecules"""
    def __init__(self,
                 spectra: Sequence[Spectrum],
                 spectrum_indexes_per_inchikey: dict[str, Iterable],
                 embeddings: Optional[Embeddings] = None,
                 progress_bars=False):
        self._spectra = tuple([spectrum.clone() for spectrum in spectra])
        self.spectrum_indexes_per_inchikey: dict[str, tuple[int]] = {key: tuple(values) for key, values in spectrum_indexes_per_inchikey.items()}
        self.progress_bars = progress_bars
        self._embeddings = embeddings

    @classmethod
    def create_spectrum_set(cls, spectra: Sequence[Spectrum], progress_bars=False) -> "AnnotatedSpectrumSet":
        spectrum_indexes_per_inchikey = defaultdict(list)
        for spectrum_index, spectrum in enumerate(spectra):
            inchikey = spectrum.get("inchikey")
            if inchikey is None:
                raise ValueError("Annotated Spectrum set expects spectra that all have an inchikey")
            spectrum_indexes_per_inchikey[inchikey[:14]].append(spectrum_index)
        return cls(spectra, spectrum_indexes_per_inchikey, progress_bars=progress_bars)

    def __add__(self, other) -> "AnnotatedSpectrumSet":
        """Adds two spectrum sets together"""
        if not isinstance(other, AnnotatedSpectrumSet):
            return NotImplemented
        spectra = self.spectra + other.spectra
        # update spectrum_indexes_per_inchikey
        starting_index = len(self.spectra)
        reindexed_indexes_per_inchikey = {}
        for inchikey, list_of_spectrum_indexes in other.spectrum_indexes_per_inchikey.items():
            reindexed_indexes_per_inchikey[inchikey] = [v + starting_index for v in list_of_spectrum_indexes]
        # combine indexes
        spectrum_indexes_per_inchikey = defaultdict(list)
        for indexes_per_inchikey in (self.spectrum_indexes_per_inchikey, reindexed_indexes_per_inchikey):
            for inchikey, indexes in indexes_per_inchikey.items():
                spectrum_indexes_per_inchikey[inchikey].extend(indexes)

        # combine embeddings
        embeddings = None
        if self._embeddings and other._embeddings:
            embeddings = Embeddings.combine_embeddings(self.embeddings, other.embeddings)
        return AnnotatedSpectrumSet(spectra,
                                    spectrum_indexes_per_inchikey,
                                    embeddings=embeddings,
                                    progress_bars=self.progress_bars)

    def subset_spectra(self, spectrum_indexes) -> "AnnotatedSpectrumSet":
        """Returns a new instance of a subset of the spectra"""
        spectra = [self._spectra[index] for index in spectrum_indexes]
        new_instance = AnnotatedSpectrumSet.create_spectrum_set(spectra, progress_bars=self.progress_bars)
        if self._embeddings is not None:
            new_instance._embeddings = self.embeddings.subset_embeddings(spectra)
        return new_instance

    def spectra_per_inchikey(self, inchikey) -> List[Spectrum]:
        matching_spectra = []
        for index in self.spectrum_indexes_per_inchikey[inchikey]:
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
            raise ValueError("First run add_embeddings")
        return self._embeddings

    @property
    def inchikeys(self):
        return tuple(self.spectrum_indexes_per_inchikey.keys())

    def __copy__(self):
        return AnnotatedSpectrumSet(self.spectra,
                                    self.spectrum_indexes_per_inchikey,
                                    self.embeddings,
                                    progress_bars=self.progress_bars)

    def __eq__(self, other: "AnnotatedSpectrumSet"):
        if not isinstance(other, AnnotatedSpectrumSet):
            raise NotImplemented
        if self.spectra != other.spectra:
            return False
        if self.spectrum_indexes_per_inchikey != other.spectrum_indexes_per_inchikey:
            return False
        if self._embeddings != other._embeddings:
            return False
        return True
