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
        factual_probs, counterfactual_probs = self.pick_probs(self.predictor, factual_events, factual_features, counterfactual_events, counterfactual_features)

        improvements = self.compute_diff(self.valuator, factual_probs, counterfactual_probs)

        self.results = improvements
        return self

    def pick_probs(self, factual_events, factual_features, counterfactual_events, counterfactual_features):
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    def compute_diff(self, valuator, factual_probs, counterfactual_probs):
        raise NotImplementedError("This function (compute_diff) needs to be implemented")

    def normalize(self):
        normed_values = self.results / self.results.sum(axis=1, keepdims=True)
        self.normalized_results = normed_values
        return self


class MultipleDiffsMixin():
    def compute_diff(self, valuator, factual_probs, counterfactual_probs):
        improvements = valuator(counterfactual_probs.prod(-1, keepdims=True), factual_probs.prod(-1, keepdims=False)).T
        return improvements


class SingularDiffsMixin():
    def compute_diff(self, valuator, factual_probs, counterfactual_probs):
        shape = factual_probs.shape
        improvements = valuator(counterfactual_probs[..., None], factual_probs.reshape(shape[:-1] + (1, shape[-1])))
        # improvements = improvements.sum(axis=-2)
        return improvements


class OutcomeMixin():
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
        super(SummarizedNextActivityImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)

class SequenceNextActivityImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceNextActivityImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)

class SummarizedOddsImprovementMeasureDiffs(SequenceMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)
 
class SequenceOddsImprovementMeasureDiffs(SequenceMixin, MultipleDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SequenceOddsImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)
 
class SummarizedOutcomeImprovementMeasureDiffs(OutcomeMixin, SingularDiffsMixin, ImprovementMeasure):
    def __init__(self, vocab_len, max_len, prediction_model: tf.keras.Model) -> None:
        super(SummarizedOutcomeImprovementMeasureDiffs, self).__init__(vocab_len, max_len, prediction_model=prediction_model, valuation_function=distances.likelihood_difference)



# TODO: Add a version for whole sequence differences

if __name__ == "__main__":
    from thesis_predictors.models.lstms.lstm import OutcomeLSTM
    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 50
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}
    # generative_reader = GenerativeDataset(reader)
    (cf_events, cf_features) = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)[0]
    (fa_events, fa_features) = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)[0]
    # fa_events[:, -2] = 8
    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    # model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects={"JoinedLoss": OutcomeLSTM.init_metrics()[1]})
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects={obj.name: obj for obj in OutcomeLSTM.init_metrics()})
    improvement_computer = SummarizedOutcomeImprovementMeasureDiffs(reader.vocab_len, reader.max_len, model)
    print(improvement_computer.compute_valuation(fa_events[1:3], fa_features[1:3], cf_events, cf_features))
