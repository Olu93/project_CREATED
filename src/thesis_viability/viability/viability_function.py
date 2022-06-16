from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Sequence

from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, ConfigurationSet, Viabilities

import itertools as it

import numpy as np
import tensorflow as tf

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

    def to_list(self):
        return np.array([self.use_sparcity, self.use_similarity, self.use_dllh, self.use_ollh])

    def to_num(self):
        return self.to_list() * 1

    def to_binstr(self) -> str:
        return "".join([str(int(val)) for val in self.to_list()])

    @staticmethod
    def get_combinations(skip_all_false: bool = True) -> Sequence[MeasureMask]:
        comb_g = it.product(*([(False, True)] * 4))
        combs = list(comb_g)
        masks = [MeasureMask(*comb) for comb in combs if not (skip_all_false and (not any(comb)))]
        return masks

    def __repr__(self):
        return repr(self.to_dict())


# TODO: Normalise
class ViabilityMeasure(ConfigurableMixin):


    def __init__(self, vocab_len: int, max_len: int, training_data: Cases, prediction_model: tf.keras.Model, measures:ConfigurationSet=None) -> None:
        self._training_data = training_data
        self.sparcity_computer = SparcityMeasure(vocab_len, max_len)
        self.similarity_computer = SimilarityMeasure(vocab_len, max_len)
        self.datalikelihood_computer = DatalikelihoodMeasure(vocab_len, max_len, training_data=self._training_data)
        self.outcomellh_computer = OutcomelikelihoodMeasure(vocab_len, max_len, prediction_model=prediction_model)
        self.measure_mask = MeasureMask()

    def get_config(self) -> BetterDict:
        return BetterDict(super().get_config()).merge({"len_training_data": len(self._training_data)})

    def apply_measure_mask(self, measure_mask: MeasureMask = None):
        self.measure_mask = measure_mask
        return self

    def compute(self, fa_cases:Cases, cf_cases:Cases, is_multiplied: bool = False) -> Viabilities:
        # fa_events, fa_features = fa_cases.cases
        # cf_events, cf_features = cf_cases.cases
        self.partial_values = {}
        res = Viabilities(len(cf_cases), len(fa_cases))
        result = 0 if not is_multiplied else 1
        if self.measure_mask.use_similarity:
            temp = self.similarity_computer.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_similarity(temp.T)
        if self.measure_mask.use_sparcity:
            temp = self.sparcity_computer.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_sparcity(temp.T)
        if self.measure_mask.use_dllh:
            temp = self.datalikelihood_computer.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_dllh(temp.T)
        if self.measure_mask.use_ollh:
            computation = self.outcomellh_computer.compute_valuation(fa_cases, cf_cases).normalize()
            temp = computation.normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_ollh(temp.T)
            res.set_mllh(computation.llh)
        res.set_viability(result.T)
        return res

    def __call__(self, fa_cases:Cases, cf_cases:Cases, is_multiplied=False) -> Viabilities:
        return self.compute(fa_cases, cf_cases, is_multiplied=is_multiplied)

