import random
from typing import Callable, List, Tuple
import numpy as np
from matchms.similarity.vector_similarity_functions import jaccard_similarity_matrix
from tqdm import tqdm
from ms2query.benchmarking.SpectrumDataSet import SpectraWithFingerprints, SpectrumSet


class EvaluateMethods:
    def __init__(
        self, training_spectrum_set: SpectraWithFingerprints, validation_spectrum_set: SpectraWithFingerprints
    ):
        self.training_spectrum_set = training_spectrum_set
        self.validation_spectrum_set = validation_spectrum_set

        self.training_spectrum_set.progress_bars = False
        self.validation_spectrum_set.progress_bars = False

    def benchmark_analogue_search(
        self,
        prediction_function: Callable[
            [SpectraWithFingerprints, SpectraWithFingerprints], Tuple[List[str], List[float]]
        ],
    ) -> float:
        predicted_inchikeys, _ = prediction_function(self.training_spectrum_set, self.validation_spectrum_set)
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
                    predicted_fingerprint = self.training_spectrum_set.inchikey_fingerprint_pairs[predicted_inchikey]
                    actual_fingerprint = self.validation_spectrum_set.inchikey_fingerprint_pairs[inchikey]
                    tanimoto_for_prediction = calculate_tanimoto_score_between_pair(
                        predicted_fingerprint, actual_fingerprint
                    )
                    prediction_scores.append(tanimoto_for_prediction)

            average_prediction = sum(prediction_scores) / len(prediction_scores)
            score = average_prediction
            average_scores_per_inchikey.append(score)
        average_over_all_inchikeys = sum(average_scores_per_inchikey) / len(average_scores_per_inchikey)
        return average_over_all_inchikeys

    def benchmark_exact_matching_within_ionmode(
        self,
        prediction_function: Callable[
            [SpectraWithFingerprints, SpectraWithFingerprints], Tuple[List[str], List[float]]
        ],
        ionmode: str,
    ) -> float:
        """Test the accuracy at retrieving exact matches from the library

        For each inchikey with more than 1 spectrum the spectra are split in two sets. Half for each inchikey is added
        to the library (training set), for the other half predictions are made. Thereby there is always an exact match
        avaialable. Only the highest ranked prediction is considered correct if the correct inchikey is predicted.
        An accuracy per inchikey is calculated followed by calculating the average.
        """
        selected_spectra = subset_spectra_on_ionmode(self.validation_spectrum_set, ionmode)

        set_1, set_2 = split_spectrum_set_per_inchikeys(selected_spectra)

        predicted_inchikeys = predict_between_two_sets(self.training_spectrum_set, set_1, set_2, prediction_function)

        # add the spectra to set_1
        set_1.add_spectra(set_2)
        return calculate_average_exact_match_accuracy(set_1, predicted_inchikeys)

    def exact_matches_across_ionization_modes(
        self,
        prediction_function: Callable[
            [SpectraWithFingerprints, SpectraWithFingerprints], Tuple[List[str], List[float]]
        ],
    ):
        """Test the accuracy at retrieving exact matches from the library if only available in other ionisation mode

        Each val spectrum is matched against the training set with the other val spectra of the same inchikey, but other
         ionisation mode added to the library.
        """
        pos_set, neg_set = split_spectrum_set_per_inchikey_across_ionmodes(self.validation_spectrum_set)
        predicted_inchikeys = predict_between_two_sets(
            self.training_spectrum_set, pos_set, neg_set, prediction_function
        )
        # add the spectra to set_1
        pos_set.add_spectra(neg_set)
        return calculate_average_exact_match_accuracy(pos_set, predicted_inchikeys)

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


def predict_between_two_sets(
    library: SpectrumSet, query_set_1: SpectrumSet, query_set_2: SpectrumSet, prediction_function
):
    """Makes predictions between query sets and the library, with the other query set added.

    This is necessary for testing exact matching"""
    training_set_copy = library.copy()
    training_set_copy.add_spectra(query_set_2)
    predicted_inchikeys_1, _ = prediction_function(training_set_copy, query_set_1)

    training_set_copy = library.copy()
    training_set_copy.add_spectra(query_set_1)
    predicted_inchikeys_2, _ = prediction_function(training_set_copy, query_set_2)

    return predicted_inchikeys_1 + predicted_inchikeys_2


def calculate_average_exact_match_accuracy(spectrum_set: SpectrumSet, predicted_inchikeys: List[str]):
    if len(spectrum_set.spectra) != len(predicted_inchikeys):
        raise ValueError("The number of spectra should be equal to the number of predicted inchikeys ")
    exact_match_accuracy_per_inchikey = []
    for inchikey in tqdm(
        spectrum_set.spectrum_indexes_per_inchikey.keys(), desc="Calculating exact match accuracy per inchikey"
    ):
        val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indexes_per_inchikey[inchikey]
        correctly_predicted = 0
        for selected_spectrum_idx in val_spectrum_indexes_matching_inchikey:
            if inchikey == predicted_inchikeys[selected_spectrum_idx]:
                correctly_predicted += 1
        exact_match_accuracy_per_inchikey.append(correctly_predicted / len(val_spectrum_indexes_matching_inchikey))
    return sum(exact_match_accuracy_per_inchikey) / len(exact_match_accuracy_per_inchikey)


def split_spectrum_set_per_inchikeys(spectrum_set: SpectrumSet) -> Tuple[SpectrumSet, SpectrumSet]:
    """Splits a spectrum set into two.
    For each inchikey with more than one spectrum the spectra are divided over the two sets"""
    indexes_set_1 = []
    indexes_set_2 = []
    for inchikey in tqdm(spectrum_set.spectrum_indexes_per_inchikey.keys(), desc="Splitting spectra per inchikey"):
        val_spectrum_indexes_matching_inchikey = spectrum_set.spectrum_indexes_per_inchikey[inchikey]
        if len(val_spectrum_indexes_matching_inchikey) == 1:
            # all single spectra are excluded from this test, since no exact match can be added to the library
            continue
        split_index = len(val_spectrum_indexes_matching_inchikey) // 2
        random.shuffle(val_spectrum_indexes_matching_inchikey)
        indexes_set_1.extend(val_spectrum_indexes_matching_inchikey[:split_index])
        indexes_set_2.extend(val_spectrum_indexes_matching_inchikey[split_index:])
    return spectrum_set.subset_spectra(indexes_set_1), spectrum_set.subset_spectra(indexes_set_2)


def split_spectrum_set_per_inchikey_across_ionmodes(
    spectrum_set: SpectrumSet,
) -> Tuple[SpectrumSet, SpectrumSet]:
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


def subset_spectra_on_ionmode(spectrum_set: SpectrumSet, ionmode) -> SpectrumSet:
    spectrum_indexes_to_keep = []
    for i, spectrum in enumerate(spectrum_set.spectra):
        if spectrum.get("ionmode") == ionmode:
            spectrum_indexes_to_keep.append(i)
    return spectrum_set.subset_spectra(spectrum_indexes_to_keep)


def calculate_tanimoto_score_between_pair(fingerprint_1: str, fingerprint_2: str) -> float:
    return jaccard_similarity_matrix(np.array([fingerprint_1]), np.array([fingerprint_2]))[0][0]
