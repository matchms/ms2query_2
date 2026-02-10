from typing import List
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

    def benchmark_analogue_search(
            self, predicted_inchikeys: List[str],
    ) -> float:
        """Calculates the average accuracy of predictions made.

        predicted_inchikeys should match the order in self.validation_spectrum_set
        """
        average_scores_per_inchikey = []
        # Calculate score per unique inchikey
        for inchikey in tqdm(
                self.validation_spectrum_set.spectrum_indices_per_inchikey.keys(),
                desc="Calculating analogue accuracy per inchikey",
        ):
            matching_spectrum_indexes = self.validation_spectrum_set.spectrum_indices_per_inchikey[inchikey]
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


def calculate_tanimoto_score_between_pair(fingerprint_1: np.ndarray, fingerprint_2: np.ndarray) -> float:
    return jaccard_similarity_matrix(np.array([fingerprint_1]), np.array([fingerprint_2]))[0][0]
