from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thesis_commons.model_commons import TensorflowModelMixin

from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

# from thesis_viability.helper.base_distances import likelihood_difference as dist
import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import Cases
from thesis_viability.helper.base_distances import BaseDistance, MeasureMixin

DEBUG = True

# TODO: Alternatively also use custom damerau_levenshtein method for data likelihood


class ImprovementMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model, valuation_function: Callable) -> None:
        super(ImprovementMeasure, self).__init__(vocab_len, max_len)
        self.predictor = prediction_model
        self.valuator = valuation_function

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> ImprovementMeasure:
        factual_probs, new_cf_probs = self.pick_probs(self.predictor, fa_cases, cf_cases)

        improvements = self.compute_diff(self.valuator, factual_probs, new_cf_probs)
        self.llh: NDArray = new_cf_probs
        self.results: NDArray = improvements
        return self

    def pick_probs(self, fa_cases: Cases, cf_cases: Cases) -> Tuple[NDArray, NDArray]:
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    def compute_diff(self, valuator:BaseDistance, original_probs:NDArray, new_probs:NDArray) -> NDArray:
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    # def normalize(self):
    #     normed_values = self.results / self.results.sum(axis=1, keepdims=True)
    #     self.normalized_results = normed_values
    #     return self

    def normalize(self):
        # normed_values = self.results / self.results.sum(axis=1, keepdims=True)
        self.normalized_results = (1 + self.results)/2
        return self


class MultipleDiffsMixin():
    def compute_diff(self, valuator, target_probs, counterfactual_probs) -> NDArray:
        improvements = valuator(counterfactual_probs.prod(-1, keepdims=True), target_probs.prod(-1, keepdims=False)).T
        return improvements


class SingularDiffsMixin():
    def compute_diff(self, valuator:BaseDistance, original_probs:NDArray, new_probs:NDArray) -> NDArray:
        original_probs.shape
        improvements = valuator(new_probs, original_probs.T)
        # improvements = improvements.sum(axis=-2)
        return improvements.T


class OutcomeMixin():
    def compute_diff(self, valuator, original_probs: NDArray, new_cf_probs: NDArray) -> NDArray:
        len_fa, len_cf = original_probs.shape[0], new_cf_probs.shape[0]
        fa_counter_probs = (1 - original_probs)  # counter probaility of factual as we want to invert those
        new_cf_probs = new_cf_probs.T
        fa_counter_outcomes = fa_counter_probs > .5 # The direction to go
        fa_expanded_counter_probs = np.repeat(fa_counter_probs, len_cf, axis=1)
        expanded_new_cf_probs = np.repeat(new_cf_probs, len_fa, axis=0)
        
        differences = np.abs(valuator(fa_expanded_counter_probs, expanded_new_cf_probs)) # General difference
        fa_is_higher = fa_expanded_counter_probs > expanded_new_cf_probs # If fa higher then True
        fa_is_on_track = ~(fa_is_higher ^ fa_counter_outcomes) # XNOR operation, where higher=True & outcome=True as well as higher=False & outcome=False result in True 
        
        diffs = np.array(differences)
        diffs[fa_is_on_track] *= -1 # Everycase in which fa is on track is a bad case. Hence, negative difference.
        
        return diffs
    

    def pick_probs(self, predictor: TensorflowModelMixin, fa_cases: Cases, cf_cases: Cases) -> Tuple[NDArray, NDArray]:
        # factual_probs = predictor.call([factual_events.astype(np.float32), factual_features.astype(np.float32)])
        factual_probs = predictor.predict(fa_cases.cases)
        counterfactual_probs = predictor.predict(cf_cases.cases)
        return factual_probs, counterfactual_probs


class SequenceMixin():
    def pick_probs(self, predictor: TensorflowModelMixin, fa_cases: Cases, cf_cases: Cases) -> Tuple[NDArray, NDArray]:
        factual_likelihoods: NDArray = predictor.predict(fa_cases.cases)
        batch, seq_len, vocab_size = factual_likelihoods.shape
        factual_probs = np.take_along_axis(factual_likelihoods.reshape(-1, vocab_size), fa_cases.events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)
        counterfactual_likelihoods: NDArray = predictor.predict(cf_cases.cases)
        batch, seq_len, vocab_size = counterfactual_likelihoods.shape
        counterfactual_probs = np.take_along_axis(counterfactual_likelihoods.reshape(-1, vocab_size), cf_cases.events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)

        # TODO: This is simplified. Should actually compute the likelihoods by picking the correct event probs iteratively
        return factual_probs, counterfactual_probs


class SummarizedNextActivityImprovementMeasureOdds(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedNextActivityImprovementMeasureOdds, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.OddsRatio())


class SummarizedNextActivityImprovementMeasureDiffs(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedNextActivityImprovementMeasureDiffs, self).__init__(vocab_len,
                                                                            max_len,
                                                                            prediction_model=prediction_model,
                                                                            valuation_function=distances.LikelihoodDifference())


class SequenceNextActivityImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceNextActivityImprovementMeasureDiffs, self).__init__(vocab_len,
                                                                          max_len,
                                                                          prediction_model=prediction_model,
                                                                          valuation_function=distances.LikelihoodDifference())


class SummarizedOddsImprovementMeasureDiffs(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


class SequenceOddsImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


class OutcomeImprovementMeasureDiffs(OutcomeMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(OutcomeImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


# TODO: Add a version for whole sequence differences
