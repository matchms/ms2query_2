import random
from typing import Callable, List, Tuple
from tqdm import tqdm
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet


class EvaluateExactMatchSearchAcrossIonmodes:
    def __init__(
        self,
        training_spectrum_set: AnnotatedSpectrumSet,
        validation_spectrum_set: AnnotatedSpectrumSet,
    ):
        self.training_spectrum_set = training_spectrum_set
        (self.pos_validation_spectra, self.neg_validation_spectra) = (
            self.split_spectrum_set_per_inchikey_across_ionmodes(validation_spectrum_set)
        )

    def pos_in_neg(
        self,
        prediction_function: Callable[[AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]],
    ):
        return get_exact_match_accuracy(
            self.pos_validation_spectra,
            prediction_function(self.training_spectrum_set + self.neg_validation_spectra, self.pos_validation_spectra),
        )

    def neg_in_pos(
        self,
        prediction_function: Callable[[AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]],
    ):
        return get_exact_match_accuracy(
            self.neg_validation_spectra,
            prediction_function(self.training_spectrum_set + self.pos_validation_spectra, self.neg_validation_spectra),
        )

    @staticmethod
    def split_spectrum_set_per_inchikey_across_ionmodes(
        spectrum_set: AnnotatedSpectrumSet,
    ) -> Tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
        """Splits a spectrum set in two sets on ionmode.
        Only uses spectra for inchikeys with at least 1 pos and 1 neg"""
        all_pos_indexes = []
        all_neg_indexes = []
        for inchikey in tqdm(
            spectrum_set.spectrum_indices_per_inchikey.keys(),
            desc="Splitting spectra per inchikey across ionmodes",
        ):
            val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indices_per_inchikey[inchikey]
            positive_val_spectrum_indexes_current_inchikey = []
            negative_val_spectrum_indexes_current_inchikey = []
            for spectrum_index in val_spectrum_indexes_matching_inchikey:
                ionmode = spectrum_set.spectra[spectrum_index].get("ionmode")
                if ionmode == "positive":
                    positive_val_spectrum_indexes_current_inchikey.append(spectrum_index)
                elif ionmode == "negative":
                    negative_val_spectrum_indexes_current_inchikey.append(spectrum_index)

            if (
                len(positive_val_spectrum_indexes_current_inchikey) < 1
                or len(negative_val_spectrum_indexes_current_inchikey) < 1
            ):
                continue
            else:
                all_pos_indexes.extend(positive_val_spectrum_indexes_current_inchikey)
                all_neg_indexes.extend(negative_val_spectrum_indexes_current_inchikey)

        pos_val_spectra = spectrum_set.subset_spectra(all_pos_indexes)
        neg_val_spectra = spectrum_set.subset_spectra(all_neg_indexes)
        return pos_val_spectra, neg_val_spectra


class EvaluateExactMatchSearchWithinIonmodes:
    def __init__(
        self,
        training_spectrum_set: AnnotatedSpectrumSet,
        validation_spectrum_set: AnnotatedSpectrumSet,
    ):
        self.training_spectrum_set = training_spectrum_set
        self.pos_split_per_inchikey_set_1, self.pos_split_per_inchikey_set_2 = self._split_spectrum_set_per_inchikeys(
            subset_spectra_on_ionmode(validation_spectrum_set, "positive")
        )

        self.neg_split_per_inchikey_set_1, self.neg_split_per_inchikey_set_2 = self._split_spectrum_set_per_inchikeys(
            subset_spectra_on_ionmode(validation_spectrum_set, "negative")
        )

    def neg_in_neg(
        self,
        prediction_function: Callable[[AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]],
    ):
        accuracy_set_2 = get_exact_match_accuracy(
            self.neg_split_per_inchikey_set_2,
            prediction_function(
                self.training_spectrum_set + self.neg_split_per_inchikey_set_1, self.neg_split_per_inchikey_set_2
            ),
        )
        accuracy_set_1 = get_exact_match_accuracy(
            self.neg_split_per_inchikey_set_1,
            prediction_function(
                self.training_spectrum_set + self.neg_split_per_inchikey_set_2, self.neg_split_per_inchikey_set_1
            ),
        )
        return (accuracy_set_2 + accuracy_set_1) / 2

    def pos_in_pos(
        self,
        prediction_function: Callable[[AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]],
    ):
        accuracy_set_2 = get_exact_match_accuracy(
            self.pos_split_per_inchikey_set_2,
            prediction_function(
                self.training_spectrum_set + self.pos_split_per_inchikey_set_1, self.pos_split_per_inchikey_set_2
            ),
        )
        accuracy_set_1 = get_exact_match_accuracy(
            self.pos_split_per_inchikey_set_1,
            prediction_function(
                self.training_spectrum_set + self.pos_split_per_inchikey_set_2, self.pos_split_per_inchikey_set_1
            ),
        )
        return (accuracy_set_2 + accuracy_set_1) / 2

    @staticmethod
    def _split_spectrum_set_per_inchikeys(
        spectrum_set: AnnotatedSpectrumSet, seed=42
    ) -> Tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
        """Splits a spectrum set into two.
        For each inchikey with more than one spectrum the spectra are divided over the two sets"""
        indexes_set_1 = []
        indexes_set_2 = []
        rng = random.Random(seed)
        for inchikey in tqdm(spectrum_set.spectrum_indices_per_inchikey.keys(), desc="Splitting spectra per inchikey"):
            val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indices_per_inchikey[inchikey]
            if len(val_spectrum_indexes_matching_inchikey) == 1:
                # all single spectra are excluded from this test, since no exact match can be added to the library
                continue
            split_index = len(val_spectrum_indexes_matching_inchikey) // 2
            rng.shuffle(list(val_spectrum_indexes_matching_inchikey))
            indexes_set_1.extend(val_spectrum_indexes_matching_inchikey[:split_index])
            indexes_set_2.extend(val_spectrum_indexes_matching_inchikey[split_index:])
        return spectrum_set.subset_spectra(indexes_set_1), spectrum_set.subset_spectra(indexes_set_2)


def get_exact_match_accuracy(query_spectrum_set, predicted_inchikeys):
    exact_match_accuracy_per_inchikey = []
    for inchikey in tqdm(query_spectrum_set.inchikeys, desc="Calculating exact match accuracy per inchikey"):
        val_spectrum_indexes_matching_inchikey = query_spectrum_set.spectrum_indexes_per_inchikey[inchikey]
        correctly_predicted = 0
        for selected_spectrum_idx in val_spectrum_indexes_matching_inchikey:
            if inchikey == predicted_inchikeys[selected_spectrum_idx]:
                correctly_predicted += 1
        exact_match_accuracy_per_inchikey.append(correctly_predicted / len(val_spectrum_indexes_matching_inchikey))
    return sum(exact_match_accuracy_per_inchikey) / len(exact_match_accuracy_per_inchikey)


def subset_spectra_on_ionmode(spectrum_set: AnnotatedSpectrumSet, ionmode) -> AnnotatedSpectrumSet:
    spectrum_indexes_to_keep = []
    for i, spectrum in enumerate(spectrum_set.spectra):
        if spectrum.get("ionmode") == ionmode:
            spectrum_indexes_to_keep.append(i)
    return spectrum_set.subset_spectra(spectrum_indexes_to_keep)
