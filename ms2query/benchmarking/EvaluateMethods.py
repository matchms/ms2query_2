import random
from typing import Callable, List, Tuple
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.AnnotatedSpectrumSet import AnnotatedSpectrumSet
from ms2query.benchmarking.Fingerprints import Fingerprints

class EvaluateAnalogueSearch:
    def __init__(
            self, training_spectrum_set: AnnotatedSpectrumSet,
            validation_spectrum_set: AnnotatedSpectrumSet,
            fingerprint_type="daylight",
            nbits=2048,
    ):
        self.training_spectrum_set = training_spectrum_set
        self.validation_spectrum_set = validation_spectrum_set

        self.fingerprints = Fingerprints.from_spectrum_set(training_spectrum_set + validation_spectrum_set,
                                                           fingerprint_type, nbits)
        self.training_spectrum_set.progress_bars = False
        self.validation_spectrum_set.progress_bars = False

    def benchmark_analogue_search(
            self, predicted_inchikeys: List[str],
    ) -> float:
        """Calculates the average accuracy of predictions made.

        predicted_inchikeys should match the order in self.validation_spectrum_set
        """
        average_scores_per_inchikey = []
        # Calculate score per unique inchikey
        for inchikey in tqdm(
                self.validation_spectrum_set.spectrum_indexes_per_inchikey.keys(),
                desc="Calculating analogue accuracy per inchikey",
        ):
            matching_spectrum_indexes = self.validation_spectrum_set.spectrum_indexes_per_inchikey[inchikey]
            prediction_scores = []
            for index in matching_spectrum_indexes:
                predicted_inchikey = predicted_inchikeys[index]
                if predicted_inchikey is None:
                    prediction_scores.append(0.0)
                else:
                    predicted_fingerprint = self.fingerprints.get_fingerprints([predicted_inchikey])[0]
                    actual_fingerprint = self.fingerprints.get_fingerprints([inchikey])[0]
                    tanimoto_for_prediction = calculate_tanimoto_score_between_pair(
                        predicted_fingerprint, actual_fingerprint
                    )
                    prediction_scores.append(tanimoto_for_prediction)

            average_prediction = sum(prediction_scores) / len(prediction_scores)
            score = average_prediction
            average_scores_per_inchikey.append(score)
        average_over_all_inchikeys = sum(average_scores_per_inchikey) / len(average_scores_per_inchikey)
        return average_over_all_inchikeys

    def get_accuracy_recall_curve(self):
        """This method should test the recall accuracy balance.
        All of the used methods use a threshold which indicates quality of prediction.
        A method that can predict well when a prediction is accurate is beneficial.
        We need a method to test this.

        One method is generating a recall accuracy curve. This could be done for both the analogue search predictions
        and the exact match predictions. By returning the predicted score for a match this method could create an
        accuracy recall plot.
        """
        raise NotImplementedError

class EvaluateExactMatchSearch:
    def __init__(
        self, training_spectrum_set: AnnotatedSpectrumSet,
            validation_spectrum_set: AnnotatedSpectrumSet,
    ):
        self.training_spectrum_set = training_spectrum_set
        (self.pos_validation_spectra,
         self.neg_validation_spectra) = split_spectrum_set_per_inchikey_across_ionmodes(validation_spectrum_set)

        self.pos_split_per_inchikey_set_1, self.pos_split_per_inchikey_set_2 = split_spectrum_set_per_inchikeys(
            self.pos_validation_spectra)

        self.neg_split_per_inchikey_set_1, self.neg_split_per_inchikey_set_2 = split_spectrum_set_per_inchikeys(
            self.neg_validation_spectra)

    def pos_in_neg(self, prediction_function: Callable[
            [AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]
        ],):
        return get_exact_match_accuracy(self.pos_validation_spectra,
                                        prediction_function(self.training_spectrum_set + self.neg_validation_spectra,
                                                            self.pos_validation_spectra))
    def neg_in_pos(self, prediction_function: Callable[
            [AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]
        ],):
        return get_exact_match_accuracy(self.neg_validation_spectra,
                                        prediction_function(self.training_spectrum_set + self.pos_validation_spectra,
                                                            self.neg_validation_spectra))

    def neg_in_neg(self, prediction_function: Callable[
            [AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]
        ],):
        accuracy_set_2 = get_exact_match_accuracy(
            self.neg_split_per_inchikey_set_2,
            prediction_function(self.training_spectrum_set + self.neg_split_per_inchikey_set_1,
                                self.neg_split_per_inchikey_set_2))
        accuracy_set_1 = get_exact_match_accuracy(
            self.neg_split_per_inchikey_set_1,
            prediction_function(self.training_spectrum_set + self.neg_split_per_inchikey_set_2,
                                self.neg_split_per_inchikey_set_1))
        return (accuracy_set_2 + accuracy_set_1) / 2

    def pos_in_pos(self, prediction_function: Callable[
            [AnnotatedSpectrumSet, AnnotatedSpectrumSet], Tuple[List[str], List[float]]
        ],):
        accuracy_set_2 = get_exact_match_accuracy(
            self.pos_split_per_inchikey_set_2,
            prediction_function(self.training_spectrum_set + self.pos_split_per_inchikey_set_1,
                                self.pos_split_per_inchikey_set_2))
        accuracy_set_1 = get_exact_match_accuracy(
            self.pos_split_per_inchikey_set_1,
            prediction_function(self.training_spectrum_set + self.pos_split_per_inchikey_set_2,
                                self.pos_split_per_inchikey_set_1))

        return (accuracy_set_2 + accuracy_set_1) / 2

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


def split_spectrum_set_per_inchikeys(spectrum_set: AnnotatedSpectrumSet,
                                     seed=42) -> Tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
    """Splits a spectrum set into two.
    For each inchikey with more than one spectrum the spectra are divided over the two sets"""
    indexes_set_1 = []
    indexes_set_2 = []
    rng = random.Random(seed)
    for inchikey in tqdm(spectrum_set.spectrum_indexes_per_inchikey.keys(), desc="Splitting spectra per inchikey"):
        val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indexes_per_inchikey[inchikey]
        if len(val_spectrum_indexes_matching_inchikey) == 1:
            # all single spectra are excluded from this test, since no exact match can be added to the library
            continue
        split_index = len(val_spectrum_indexes_matching_inchikey) // 2
        rng.shuffle(list(val_spectrum_indexes_matching_inchikey))
        indexes_set_1.extend(val_spectrum_indexes_matching_inchikey[:split_index])
        indexes_set_2.extend(val_spectrum_indexes_matching_inchikey[split_index:])
    return spectrum_set.subset_spectra(indexes_set_1), spectrum_set.subset_spectra(indexes_set_2)


def split_spectrum_set_per_inchikey_across_ionmodes(
    spectrum_set: AnnotatedSpectrumSet,
) -> Tuple[AnnotatedSpectrumSet, AnnotatedSpectrumSet]:
    """Splits a spectrum set in two sets on ionmode. Only uses spectra for inchikeys with at least 1 pos and 1 neg"""
    all_pos_indexes = []
    all_neg_indexes = []
    for inchikey in tqdm(
        spectrum_set.spectrum_indexes_per_inchikey.keys(),
        desc="Splitting spectra per inchikey across ionmodes",
    ):
        val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indexes_per_inchikey[inchikey]
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


def subset_spectra_on_ionmode(spectrum_set: AnnotatedSpectrumSet, ionmode) -> AnnotatedSpectrumSet:
    spectrum_indexes_to_keep = []
    for i, spectrum in enumerate(spectrum_set.spectra):
        if spectrum.get("ionmode") == ionmode:
            spectrum_indexes_to_keep.append(i)
    return spectrum_set.subset_spectra(spectrum_indexes_to_keep)


def calculate_tanimoto_score_between_pair(fingerprint_1: np.ndarray, fingerprint_2: np.ndarray) -> float:
    return jaccard_similarity_matrix(np.array([fingerprint_1]), np.array([fingerprint_2]))[0][0]
