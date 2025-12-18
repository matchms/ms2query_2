from typing import Iterable

import numpy as np
import pandas as pd
from matchms.filtering.metadata_processing.add_fingerprint import _derive_fingerprint_from_inchi
from tqdm import tqdm

from ms2query.metrics import generalized_tanimoto_similarity_matrix


class Fingerprints:
    # I just realize that there already exists a Fingerprints class in matchms, with almost the same functionality,
    # so it will be good to merge both. The matchms version works slighlty different.
    def __init__(self, most_common_inchi_per_inchikey, fingerprint_type, nbits):
        self.most_common_inchi_per_inchikey = most_common_inchi_per_inchikey
        self.fingerprint_type = fingerprint_type
        self.nbits = nbits

        self._index_to_inchikey =np.array([], dtype="U14")
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
        self._index_to_inchikey = np.append(self._index_to_inchikey, inchikeys_to_add)
        self._inchikey_to_index = {inchikey: index for index, inchikey in enumerate(self.index_to_inchikey)}


def get_similarity_matrix(fingerprints_1: Fingerprints, fingerprints_2: Fingerprints):
    similarity_scores = generalized_tanimoto_similarity_matrix(fingerprints_1.fingerprints, fingerprints_2.fingerprints)
    return pd.DataFrame(similarity_scores, index=fingerprints_1.index_to_inchikey, columns=fingerprints_2.index_to_inchikey)
