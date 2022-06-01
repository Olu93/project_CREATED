import glob
import io
import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

import thesis_commons.metric as metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers import OutcomeBPIC12Reader as Reader
from thesis_viability.feasibility.feasibility_metric import FeasibilityMeasure
from thesis_viability.likelihood.likelihood_improvement import \
    OutcomeImprovementMeasureDiffs as ImprovementMeasure
from thesis_viability.similarity.similarity_metric import SimilarityMeasure
from thesis_viability.sparcity.sparcity_metric import SparcityMeasure

DEBUG = True


# TODO: Normalise
class ViabilityMeasure:
    SPARCITY = 0
    SIMILARITY = 1
    FEASIBILITY = 2
    IMPROVEMENT = 3

    def __init__(self, vocab_len, max_len, training_data, prediction_model) -> None:
        tr_events, tr_features = training_data
        self.sparcity_computer = SparcityMeasure(vocab_len, max_len)
        self.similarity_computer = SimilarityMeasure(vocab_len, max_len)
        self.feasibility_computer = FeasibilityMeasure(vocab_len, max_len, training_data=training_data)
        self.improvement_computer = ImprovementMeasure(vocab_len, max_len, prediction_model=prediction_model)
        self.partial_values = None
        

    def compute_valuation(self, fa_events, fa_features, cf_events, cf_features, fa_outcomes=None, is_multiplied=False):
        feasibility_values = self.feasibility_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        improvement_values = self.improvement_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        sparcity_values = self.sparcity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        similarity_values = self.similarity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        # normed_feasibility_values = self.feasibility_computer.results
        # normed_improvement_values = self.improvement_computer.results

        self.partial_values = np.stack([sparcity_values, similarity_values, feasibility_values, improvement_values])

        if not is_multiplied:
            result = sparcity_values + similarity_values + feasibility_values + improvement_values
        else:
            result = sparcity_values * similarity_values * feasibility_values * improvement_values

        return result

    def __call__(self, fa_events, fa_features, cf_events, cf_features, fa_outcomes=None, is_multiplied=False) -> Any:
        return self.compute_valuation(fa_events, fa_features, cf_events, cf_features, fa_outcomes, is_multiplied=is_multiplied)

    @property
    def parts(self):
        if self.partial_values is None:
            raise ValueError("Partial values need to be computed first. Run compute_valuation!")
        return {
            'sparcity': self.partial_values[ViabilityMeasure.SPARCITY],
            'similarity': self.partial_values[ViabilityMeasure.SIMILARITY],
            # 'normed_feasibility': self.partial_values[ViabilityMeasure.NORMED_FEASIBILITY],
            # 'normed_improvement': self.partial_values[ViabilityMeasure.NORMED_IMPROVEMENT],
            'feasibility': self.partial_values[ViabilityMeasure.FEASIBILITY],
            'improvement': self.partial_values[ViabilityMeasure.IMPROVEMENT],
        }


if __name__ == "__main__":
    from thesis_predictors.models.lstms.lstm import OutcomeLSTM

    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 50
    reader = Reader(mode=task_mode).init_meta()
    custom_objects = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)

    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[0], custom_objects=custom_objects)

    viability = ViabilityMeasure(reader.vocab_len, reader.max_len, (tr_events, tr_features), model)

    viability_values = viability(fa_events, fa_features, cf_events, cf_features, fa_labels)
    print(viability_values)
    print("DONE")