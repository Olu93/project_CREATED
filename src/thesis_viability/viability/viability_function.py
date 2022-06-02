from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

import thesis_commons.metric as metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases
from thesis_readers import OutcomeBPIC12Reader as Reader
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
from thesis_viability.outcomellh.outcomllh_measure import \
    OutcomeImprovementMeasureDiffs as OutcomelikelihoodMeasure
from thesis_viability.similarity.similarity_measure import SimilarityMeasure
from thesis_viability.sparcity.sparcity_measure import SparcityMeasure

DEBUG = True


class MeasureMask:
    def __init__(self, use_sparcity: bool = True, use_similarity: bool = True, use_dllh: bool = True, use_ollh: bool = True) -> None:
        self.use_sparcity = use_sparcity
        self.use_similarity = use_similarity
        self.use_dllh = use_dllh
        self.use_ollh = use_ollh

    def to_dict(self):
        return {
            "use_sparcity": self.use_sparcity,
            "use_similarity": self.use_similarity,
            "use_dllh": self.use_dllh,
            "use_ollh": self.use_ollh,
        }


# TODO: Normalise
class ViabilityMeasure:
    SPARCITY = "sparcity"
    SIMILARITY = "similarity"
    DLLH = "dllh"
    OLLH = "ollh"

    def __init__(self, vocab_len: int, max_len: int, training_data: Cases, prediction_model: tf.keras.Model) -> None:
        self.sparcity_computer = SparcityMeasure(vocab_len, max_len)
        self.similarity_computer = SimilarityMeasure(vocab_len, max_len)
        self.datalikelihood_computer = DatalikelihoodMeasure(vocab_len, max_len, training_data=training_data)
        self.outcomellh_computer = OutcomelikelihoodMeasure(vocab_len, max_len, prediction_model=prediction_model)
        self.partial_values = {}
        self.measure_mask = MeasureMask()

    # def set_sparcity_computer(self, measure: SparcityMeasure = None):
    #     self.sparcity_computer = measure
    #     return self

    # def set_similarity_computer(self, measure: SimilarityMeasure = None):
    #     self.similarity_computer = measure
    #     return self

    # def set_dllh_computer(self, measure: DatalikelihoodMeasure = None):
    #     self.datalikelihood_computer = measure
    #     return self

    # def set_ollh_computer(self, measure: OutcomelikelihoodMeasure = None):
    #     self.outcomellh_computer = measure
    #     return self

    def set_measure_mask(self, measure_mask: MeasureMask = None):
        self.measure_mask = measure_mask
        return self

    def compute_valuation(self, fa_events, fa_features, cf_events, cf_features, is_multiplied: bool = False):
        result = 0 if not is_multiplied else 1
        if self.measure_mask.use_similarity:
            temp = self.similarity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            self.update_parts(ViabilityMeasure.SIMILARITY, temp)
        if self.measure_mask.use_sparcity:
            temp = self.sparcity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            self.update_parts(ViabilityMeasure.SPARCITY, temp)
        if self.measure_mask.use_dllh:
            temp = self.datalikelihood_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            self.update_parts(ViabilityMeasure.DLLH, temp)
        if self.measure_mask.use_ollh:
            temp = self.outcomellh_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            self.update_parts(ViabilityMeasure.OLLH, temp)

        return result

    def __call__(self, fa_events, fa_features, cf_events, cf_features, is_multiplied=False) -> Any:
        return self.compute_valuation(fa_events, fa_features, cf_events, cf_features, is_multiplied=is_multiplied)

    def update_parts(self, key, val):
        self.partial_values[key] = val
        return self

    @property
    def parts(self):
        return self.partial_values