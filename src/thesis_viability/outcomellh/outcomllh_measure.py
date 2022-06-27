from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thesis_commons.model_commons import TensorflowModelMixin

from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
# from numpy.typing import np.ndarray

# from thesis_viability.helper.base_distances import likelihood_difference as dist
import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import BetterDict, Cases
from thesis_viability.helper.base_distances import BaseDistance, MeasureMixin

DEBUG = True

# TODO: Alternatively also use custom damerau_levenshtein method for data likelihood


class ImprovementMeasure(MeasureMixin):
    prediction_model:TensorflowModelMixin = None
    evaluation_function:distances.BaseDistance = None
    def init(self, **kwargs) -> ImprovementMeasure:
        super().init(**kwargs)
        if (not self.prediction_model) or (not self.evaluation_function):
            raise Exception(f"Configuration is missing: prediction_model={self.prediction_model} evaluation_function={self.evaluation_function}")
        return self

    def set_predictor(self, prediction_model: TensorflowModelMixin) -> ImprovementMeasure:
        self.prediction_model = prediction_model
        return self

    def set_evaluator(self, evaluation_function: Callable) -> ImprovementMeasure:
        self.evaluation_function = evaluation_function
        return self

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> ImprovementMeasure:
        factual_probs, new_cf_probs = self._pick_probs(fa_cases, cf_cases)
        improvements = self._compute_diff(factual_probs, new_cf_probs)
        self.llh: np.ndarray = new_cf_probs
        self.results: np.ndarray = improvements
        return self

    def _pick_probs(self, fa_cases: Cases, cf_cases: Cases) -> Tuple[np.ndarray, np.ndarray]:
        # factual_probs = predictor.call([factual_events.astype(np.float32), factual_features.astype(np.float32)])
        factual_probs = self.prediction_model.predict(fa_cases.cases)
        counterfactual_probs = self.prediction_model.predict(cf_cases.cases)
        return factual_probs, counterfactual_probs

    def _compute_diff(self, original_probs: np.ndarray, new_cf_probs: np.ndarray) -> np.ndarray:
        len_fa, len_cf = original_probs.shape[0], new_cf_probs.shape[0]
        fa_counter_probs = (1 - original_probs)  # counter probaility of factual as we want to invert those
        new_cf_probs = new_cf_probs.T
        fa_counter_outcomes = fa_counter_probs > .5  # The direction to go
        fa_expanded_counter_probs = np.repeat(fa_counter_probs, len_cf, axis=1)
        expanded_new_cf_probs = np.repeat(new_cf_probs, len_fa, axis=0)

        differences = np.abs(self.evaluation_function(fa_expanded_counter_probs, expanded_new_cf_probs))  # General difference
        fa_is_higher = fa_expanded_counter_probs > expanded_new_cf_probs  # If fa higher then True
        fa_is_on_track = ~(fa_is_higher ^ fa_counter_outcomes)  # XNOR operation, where higher=True & outcome=True as well as higher=False & outcome=False result in True

        diffs = np.array(differences)
        diffs[fa_is_on_track] *= -1  # Everycase in which fa is on track and closer at it, is a bad case. Hence, negative difference.

        return diffs

    def normalize(self):
        # normed_values = self.results / self.results.sum(axis=1, keepdims=True)
        self.normalized_results = (1 + self.results) / 2
        return self

    def get_config(self) -> BetterDict:
        return super().get_config().merge({"type":type(self).__name__})

# class Differ(ABC):
#     def set_evaluator(self, evaluator: BaseDistance) -> Differ:
#         self.evaluator = evaluator
#         return self

#     @abstractmethod
#     def compute_diff(self, original_probs: np.ndarray, counterfactual_probs: np.ndarray) -> np.ndarray:
#         pass


# class MultipleDiffsMixin(Differ):
#     def compute_diff(self, original_probs: np.ndarray, counterfactual_probs: np.ndarray) -> np.ndarray:
#         improvements = self.evaluator(counterfactual_probs.prod(-1, keepdims=True), original_probs.prod(-1, keepdims=False)).T
#         return improvements


# class SingularDiffsMixin(Differ):
#     def compute_diff(self, original_probs: np.ndarray, counterfactual_probs: np.ndarray) -> np.ndarray:
#         improvements = self.evaluator(counterfactual_probs, original_probs.T)
#         return improvements.T


# class OutcomeDiffsMixin(Differ):
#     def compute_diff(self, original_probs: np.ndarray, new_cf_probs: np.ndarray) -> np.ndarray:
#         len_fa, len_cf = original_probs.shape[0], new_cf_probs.shape[0]
#         fa_counter_probs = (1 - original_probs)  # counter probaility of factual as we want to invert those
#         new_cf_probs = new_cf_probs.T
#         fa_counter_outcomes = fa_counter_probs > .5  # The direction to go
#         fa_expanded_counter_probs = np.repeat(fa_counter_probs, len_cf, axis=1)
#         expanded_new_cf_probs = np.repeat(new_cf_probs, len_fa, axis=0)

#         differences = np.abs(self.evaluator(fa_expanded_counter_probs, expanded_new_cf_probs))  # General difference
#         fa_is_higher = fa_expanded_counter_probs > expanded_new_cf_probs  # If fa higher then True
#         fa_is_on_track = ~(fa_is_higher ^ fa_counter_outcomes)  # XNOR operation, where higher=True & outcome=True as well as higher=False & outcome=False result in True

#         diffs = np.array(differences)
#         diffs[fa_is_on_track] *= -1  # Everycase in which fa is on track is a bad case. Hence, negative difference.

#         return diffs


# class Picker(ABC):
#     @abstractmethod
#     def pick_probs(self, original_probs: np.ndarray, new_cf_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         pass


# class OutcomePicker(Picker):
#     def pick_probs(self, predictor: TensorflowModelMixin, fa_cases: Cases, cf_cases: Cases) -> Tuple[np.ndarray, np.ndarray]:
#         # factual_probs = predictor.call([factual_events.astype(np.float32), factual_features.astype(np.float32)])
#         factual_probs = predictor.predict(fa_cases.cases)
#         counterfactual_probs = predictor.predict(cf_cases.cases)
#         return factual_probs, counterfactual_probs


# class SequentialPicker(Picker):
#     def pick_probs(self, predictor: TensorflowModelMixin, fa_cases: Cases, cf_cases: Cases) -> Tuple[np.ndarray, np.ndarray]:
#         factual_likelihoods: np.ndarray = predictor.predict(fa_cases.cases)
#         batch, seq_len, vocab_size = factual_likelihoods.shape
#         factual_probs = np.take_along_axis(factual_likelihoods.reshape(-1, vocab_size), fa_cases.events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)
#         counterfactual_likelihoods: np.ndarray = predictor.predict(cf_cases.cases)
#         batch, seq_len, vocab_size = counterfactual_likelihoods.shape
#         counterfactual_probs = np.take_along_axis(counterfactual_likelihoods.reshape(-1, vocab_size), cf_cases.events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)

#         # TODO: This is simplified. Should actually compute the likelihoods by picking the correct event probs iteratively
#         return factual_probs, counterfactual_probs


# class SummarizedNextActivityImprovementMeasureOdds(SequentialPicker, SingularDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(SummarizedNextActivityImprovementMeasureOdds, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.OddsRatio())


# class SummarizedNextActivityImprovementMeasureDiffs(SequentialPicker, SingularDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(SummarizedNextActivityImprovementMeasureDiffs, self).__init__(vocab_len,
#                                                                             max_len,
#                                                                             prediction_model=prediction_model,
#                                                                             valuation_function=distances.LikelihoodDifference())


# class SequenceNextActivityImprovementMeasureDiffs(SequentialPicker, MultipleDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(SequenceNextActivityImprovementMeasureDiffs, self).__init__(vocab_len,
#                                                                           max_len,
#                                                                           prediction_model=prediction_model,
#                                                                           valuation_function=distances.LikelihoodDifference())


# class SummarizedOddsImprovementMeasureDiffs(SequentialPicker, SingularDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(SummarizedOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


# class SequenceOddsImprovementMeasureDiffs(SequentialPicker, MultipleDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(SequenceOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


# class OutcomeImprovementMeasureDiffs(OutcomeDiffsMixin, ImprovementMeasure):
#     def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
#         super(OutcomeImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.LikelihoodDifference())


# TODO: Add a version for whole sequence differences
