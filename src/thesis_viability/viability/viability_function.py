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
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
from thesis_viability.outcomellh.outcomllh_measure import \
    OutcomeImprovementMeasureDiffs as OutcomelikelihoodMeasure
from thesis_viability.similarity.similarity_measure import SimilarityMeasure
from thesis_viability.sparcity.sparcity_measure import SparcityMeasure

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
        self.datalikelihood_computer = DatalikelihoodMeasure(vocab_len, max_len, training_data=training_data)
        self.outcomellh_computer = OutcomelikelihoodMeasure(vocab_len, max_len, prediction_model=prediction_model)
        self.partial_values = None
        

    # def set_sparcity_computer(self, measure:SparcityMeasure = None):
    #     self.sparcity_computer = measure
    # def set_similarity_computer(self, measure:SimilarityMeasure = None):
    #     self.sparcity_computer = measure
    # def set_feasibility_computer(self, measure:FeasibilityMeasure = None):
    #     self.sparcity_computer = measure
    # def set_improvement(self, measure:ImprovementMeasure = None):
    #     self.sparcity_computer = measure

    def compute_valuation(self, fa_events, fa_features, cf_events, cf_features, fa_outcomes=None, is_multiplied=False):
        datallh_values = self.datalikelihood_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        outcomellh_values = self.outcomellh_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        sparcity_values = self.sparcity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
        similarity_values = self.similarity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results

        self.partial_values = np.stack([sparcity_values, similarity_values, datallh_values, outcomellh_values])

        if not is_multiplied:
            result = sparcity_values + similarity_values + datallh_values + outcomellh_values
        else:
            result = sparcity_values * similarity_values * datallh_values * outcomellh_values

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
            'datalikelihood': self.partial_values[ViabilityMeasure.FEASIBILITY],
            'improvement': self.partial_values[ViabilityMeasure.IMPROVEMENT],
        }

