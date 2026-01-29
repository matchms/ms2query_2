from collections import Counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi
from tqdm import tqdm

from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.metrics import generalized_tanimoto_similarity_matrix


class Fingerprints:
    # I just realize that there already exists a Fingerprints class in matchms, with almost the same functionality,
    # so it will be good to merge both. The matchms version works slighlty different.
    def __init__(self, fingerprints: NDArray, inchikeys: tuple[str, ...], fingerprint_type):
        # self.most_common_inchi_per_inchikey = most_common_inchi_per_inchikey
        self.fingerprint_type = fingerprint_type
        if fingerprints.shape[0] != len(inchikeys):
            raise ValueError("The fingerprint dimension does not match the number of inchikeys")
        self._fingerprints = fingerprints
        self._inchikeys = inchikeys
        self._inchikey_to_index = {inchikey: index for index, inchikey in enumerate(self.inchikeys)}

    @classmethod
    def compute_fingerprints_from_inchi(cls, most_common_inchi_per_inchikey: dict[str, str], fingerprint_type, nbits):
        index_to_inchikey = tuple(most_common_inchi_per_inchikey.keys())
        fingerprints = np.zeros((len(index_to_inchikey), nbits), dtype=np.uint8)
        for inchikey_index, inchikey in tqdm(enumerate(index_to_inchikey), desc="Adding fingerprints to Inchikeys"):
            fingerprint = _derive_fingerprint_from_inchi(
                most_common_inchi_per_inchikey[inchikey], fingerprint_type=fingerprint_type, nbits=nbits)
            if not isinstance(fingerprint, np.ndarray):
                raise ValueError(f"Fingerprint could not be set for InChI: {most_common_inchi_per_inchikey[inchikey]}")
            fingerprints[inchikey_index, :] = fingerprint
        return cls(fingerprints, index_to_inchikey, fingerprint_type)

    @classmethod
    def from_spectrum_set(cls, spectrum_set: AnnotatedSpectrumSet, fingerprint_type, nbits):
        most_common_inchi_per_inchikey = {}
        for inchikey, spectrum_indexes in tqdm(spectrum_set.spectrum_indexes_per_inchikey.items(), desc="Get most common inchi per inchikey"):
            spectra_matching_inchikey = [spectrum_set.spectra[index] for index in spectrum_indexes]
            most_common_inchi = Counter([spectrum.get("inchi") for spectrum in spectra_matching_inchikey]).most_common(1)[0][0]
            most_common_inchi_per_inchikey[inchikey] = most_common_inchi
        return cls.compute_fingerprints_from_inchi(most_common_inchi_per_inchikey, fingerprint_type, nbits)

    @classmethod
    def combine_fingerprints(cls, fingerprints_1: "Fingerprints",
                             fingerprints_2: "Fingerprints", allow_replacing: bool=False) -> "Fingerprints":
        """Combines two sets of Fingerprints into a new instance

        allow_replacing:
            If True the fingerprints in fingerprints_1 will be replaced by fingerprints_2.
        """
        if fingerprints_1.fingerprint_type != fingerprints_2.fingerprint_type:
            raise ValueError("The fingerprint type of both Fingerprints do not match")
        if fingerprints_1.nbits != fingerprints_2.nbits:
            raise ValueError(f"The nbits for the fingerprints do not match: {fingerprints_1.nbits} != {fingerprints_2.nbits}")
        # handle overlap
        overlapping_inchikeys = set(fingerprints_2.inchikeys) & set(fingerprints_1.inchikeys)
        new_fingerprints = fingerprints_1.fingerprints.copy()
        if not np.array_equal(fingerprints_1.get_fingerprints(overlapping_inchikeys), fingerprints_2.get_fingerprints(overlapping_inchikeys)):
            if allow_replacing:
                for inchikey in overlapping_inchikeys:
                    inchikey_index = fingerprints_1._inchikey_to_index[inchikey]
                    new_fingerprints[inchikey_index, :] = fingerprints_2.fingerprints[
                                                          fingerprints_2._inchikey_to_index[inchikey], :]
            else:
                raise ValueError("The overlapping inchikeys do not have the same fingerprints")
        inchikeys_to_add = [inchikey for inchikey in fingerprints_2.inchikeys if inchikey not in fingerprints_1.inchikeys]
        combined_fingerprints = np.vstack([new_fingerprints, fingerprints_2.get_fingerprints(inchikeys_to_add)])
        combined_inchikeys = fingerprints_1.inchikeys + tuple(inchikeys_to_add)
        return cls(combined_fingerprints, combined_inchikeys, fingerprints_1.fingerprint_type)

    def get_fingerprints(self, list_of_inchikeys: Iterable[str]):
        if not isinstance(list_of_inchikeys, Iterable):
            raise TypeError("Iterable of inchikeys is expected")
        list_of_indexes = [self._inchikey_to_index[inchikey] for inchikey in list_of_inchikeys]
        return self.fingerprints[list_of_indexes]

    def subset_fingerprints(self, list_of_inchikeys) -> "Fingerprints":
        return Fingerprints(self.get_fingerprints(list_of_inchikeys), list_of_inchikeys, fingerprint_type=self.fingerprint_type)

    @property
    def inchikeys(self):
        return self._inchikeys

    @property
    def fingerprints(self):
        return self._fingerprints.view()

    @property
    def nbits(self):
        """The number of bits used"""
        return self._fingerprints.shape[1]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Fingerprints):
            return NotImplemented
        if self.fingerprint_type != other.fingerprint_type:
            print("Fingerprint type mismatch")
            return False
        if not np.array_equal(self.inchikeys, other.inchikeys):
            print("Inchikeys are not equal")
            return False
        return np.array_equal(self.fingerprints, other.fingerprints)

def get_similarity_matrix(fingerprints_1: Fingerprints, fingerprints_2: Fingerprints):
    similarity_scores = generalized_tanimoto_similarity_matrix(fingerprints_1.fingerprints, fingerprints_2.fingerprints)
    return pd.DataFrame(similarity_scores, index=fingerprints_1.inchikeys, columns=fingerprints_2.inchikeys)
