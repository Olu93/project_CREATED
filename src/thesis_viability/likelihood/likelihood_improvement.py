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
from thesis_readers import MockReader as Reader
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
        factual_likelihoods = self.predictor.predict([factual_events, factual_features])
        batch, seq_len, vocab_size = factual_likelihoods.shape
        factual_probs = np.take_along_axis(factual_likelihoods.reshape(-1, vocab_size), factual_events.astype(int).reshape(-1, 1), axis=-1).reshape(batch, seq_len)
        counterfactual_likelihoods = self.predictor.predict([counterfactual_events.astype(np.float32), counterfactual_features])
        batch, seq_len, vocab_size = counterfactual_likelihoods.shape
        counterfactual_probs = np.take_along_axis(counterfactual_likelihoods.reshape(-1, vocab_size), counterfactual_events.astype(int).reshape(-1, 1),
                                                  axis=-1).reshape(batch, seq_len)
        # TODO: This is simplified. Should actually compute the likelihoods by picking the correct event probs iteratively

        improvements = self.valuator(counterfactual_probs.prod(-1, keepdims=True), factual_probs.prod(-1, keepdims=False)).T

        self.results = improvements
        return self

    def normalize(self):
        normed_values = self.results / self.results.sum(axis=1, keepdims=True)
        self.normalized_result = normed_values
        return self


class ImprovementMeasureOdds(ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(ImprovementMeasureOdds, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.odds_ratio)


class ImprovementMeasureDiffs(ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(ImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = Reader(mode=task_mode).init_meta()
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}
    # generative_reader = GenerativeDataset(reader)
    (cf_events, cf_features) = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)[0]
    (fa_events, fa_features) = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)[0]
    # fa_events[:, -2] = 8
    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects)
    improvement_computer = ImprovementMeasure(model, distances.odds_ratio)
    print(improvement_computer.compute_valuation(fa_events[1:3], fa_features[1:3], cf_events, cf_features))
