import io
import os
from typing import Any, Callable
import numpy as np
from thesis_viability.helper.base_distances import MeasureMixin
# from thesis_viability.helper.base_distances import likelihood_difference as dist
import thesis_viability.helper.base_distances as distances
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.libcuts import layers, K, losses
import thesis_commons.metric as metric
from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset

from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
import tensorflow as tf
import pandas as pd
import glob

DEBUG = True

# TODO: Alternatively also use custom damerau_levenshtein method for data likelihood


class ImprovementMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model, valuation_function: Callable) -> None:
        super(ImprovementMeasure, self).__init__(vocab_len, max_len)
        self.predictor = prediction_model
        self.valuator = valuation_function

    def compute_valuation(self, factual_events, factual_features, counterfactual_events, counterfactual_features):
        factual_probs, new_cf_probs = self.pick_probs(self.predictor, factual_events, factual_features, counterfactual_events, counterfactual_features)

        improvements = self.compute_diff(self.valuator, factual_probs, new_cf_probs)

        self.results = improvements
        return self

    def pick_probs(self, factual_events, factual_features, counterfactual_events, counterfactual_features):
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    def compute_diff(self, valuator, factual_probs, counterfactual_probs):
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    # def normalize(self):
    #     normed_values = self.results / self.results.sum(axis=1, keepdims=True)
    #     self.normalized_results = normed_values
    #     return self

    def normalize(self):
        # normed_values = self.results / self.results.sum(axis=1, keepdims=True)
        self.normalized_results = self.results
        return self


class MultipleDiffsMixin():
    def compute_diff(self, valuator, target_probs, counterfactual_probs):
        improvements = valuator(counterfactual_probs.prod(-1, keepdims=True), target_probs.prod(-1, keepdims=False)).T
        return improvements


class SingularDiffsMixin():
    def compute_diff(self, valuator, original_probs, new_probs):
        shape = original_probs.shape
        improvements = valuator(new_probs, original_probs.T)
        # improvements = improvements.sum(axis=-2)
        return improvements.T


class OutcomeMixin():
    def compute_diff(self, valuator, original_probs, new_cf_probs):
        len_fa, len_cf = original_probs.shape[0], new_cf_probs.shape[0]
        orig_cf_probs = (1 - original_probs)
        new_cf_probs = new_cf_probs.T
        orig_cf_outcomes = orig_cf_probs > .5
        new_cf_outcomes = new_cf_probs > .5
        incompatible_outcome_mask = orig_cf_outcomes != new_cf_outcomes
        expanded_orig_cf_probs = np.repeat(orig_cf_probs, len_cf, axis=1)
        expanded_new_cf_probs = np.repeat(new_cf_probs, len_fa, axis=0)
        expanded_new_cf_probs[incompatible_outcome_mask] = 1 - expanded_new_cf_probs[incompatible_outcome_mask]

        improvements = valuator(expanded_orig_cf_probs, expanded_new_cf_probs)
        return improvements

    def pick_probs(self, predictor, factual_events, factual_features, counterfactual_events, counterfactual_features):
        # factual_probs = predictor.call([factual_events.astype(np.float32), factual_features.astype(np.float32)])
        factual_probs = predictor.predict((factual_events, factual_features))
        counterfactual_probs = predictor.predict([counterfactual_events.astype(np.float32), counterfactual_features])
        return factual_probs, counterfactual_probs


class SequenceMixin():
    def pick_probs(self, predictor, factual_events, factual_features, counterfactual_events, counterfactual_features):
        factual_likelihoods = predictor.predict((factual_events, factual_features))
        batch, seq_len, vocab_size = factual_likelihoods.shape
        factual_probs = np.take_along_axis(factual_likelihoods.reshape(-1, vocab_size), factual_events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)
        counterfactual_likelihoods = predictor.predict([counterfactual_events.astype(np.float32), counterfactual_features])
        batch, seq_len, vocab_size = counterfactual_likelihoods.shape
        counterfactual_probs = np.take_along_axis(counterfactual_likelihoods.reshape(-1, vocab_size), counterfactual_events.astype(int).reshape(-1, 1),
                                                  axis=-1).reshape(batch, seq_len)

        # TODO: This is simplified. Should actually compute the likelihoods by picking the correct event probs iteratively
        return factual_probs, counterfactual_probs


class SummarizedNextActivityImprovementMeasureOdds(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedNextActivityImprovementMeasureOdds, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.odds_ratio)


class SummarizedNextActivityImprovementMeasureDiffs(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedNextActivityImprovementMeasureDiffs, self).__init__(vocab_len,
                                                                            max_len,
                                                                            prediction_model=prediction_model,
                                                                            valuation_function=distances.likelihood_difference)


class SequenceNextActivityImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceNextActivityImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)


class SummarizedOddsImprovementMeasureDiffs(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)


class SequenceOddsImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)


class OutcomeImprovementMeasureDiffs(OutcomeMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(OutcomeImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)


# TODO: Add a version for whole sequence differences

if __name__ == "__main__":
    from thesis_predictors.models.lstms.lstm import OutcomeLSTM
    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 50
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}
    # generative_reader = GenerativeDataset(reader)
    (cf_events, cf_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    (fa_events, fa_features), y_labels, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)
    # fa_events[:, -2] = 8
    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    # model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects={"JoinedLoss": OutcomeLSTM.init_metrics()[1]})
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects={obj.name: obj for obj in OutcomeLSTM.init_metrics()})
    improvement_computer = OutcomeImprovementMeasureDiffs(reader.vocab_len, reader.max_len, model)
    print(improvement_computer.compute_valuation(fa_events[:3], fa_features[:3], cf_events[:4], cf_features[:4]).normalize())
