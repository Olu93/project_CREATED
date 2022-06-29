from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Sequence
from thesis_commons.distributions import DataDistribution
if TYPE_CHECKING:
    from thesis_commons.model_commons import TensorflowModelMixin

from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, ConfigurationSet, Viabilities

import itertools as it

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K, losses, metrics, utils, layers, optimizers, models
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
from thesis_viability.outcomellh.outcomllh_measure import ImprovementMeasure as OutcomelikelihoodMeasure
from thesis_viability.similarity.similarity_measure import SimilarityMeasure
from thesis_viability.sparcity.sparcity_measure import SparcityMeasure
import thesis_viability.helper.base_distances as distances

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


class MeasureConfig(ConfigurationSet):
    def __init__(
        self,
        sparcity: SparcityMeasure,
        similarity: SimilarityMeasure,
        dllh: DatalikelihoodMeasure,
        ollh: OutcomelikelihoodMeasure,
    ):
        self.sparcity = sparcity
        self.similarity = similarity
        self.dllh = dllh
        self.ollh = ollh
        self._list: List[MeasureConfig] = [sparcity, similarity, dllh, ollh]

    @staticmethod
    def registry(sparcity: List[SparcityMeasure] = None,
                 similarity: List[SimilarityMeasure] = None,
                 dllh: List[DatalikelihoodMeasure] = None,
                 ollh: List[OutcomelikelihoodMeasure] = None,
                 **kwargs) -> MeasureConfig:
        sparcity = sparcity or [SparcityMeasure()]
        similarity = similarity or [SimilarityMeasure()]
        dllh = dllh or [DatalikelihoodMeasure()]
        ollh = ollh or [
            OutcomelikelihoodMeasure().set_evaluator(distances.LikelihoodDifference()),
            OutcomelikelihoodMeasure().set_evaluator(distances.OddsRatio()),
        ]
        combos = it.product(sparcity, similarity, dllh, ollh)
        result = [MeasureConfig(*cnf) for cnf in combos]
        return result

    def set_vocab_len(self, vocab_len: int, **kwargs) -> MeasureConfig:
        for measure in self._list:
            measure.set_vocab_len(vocab_len)
        return self

    def set_max_len(self, max_len: int, **kwargs) -> MeasureConfig:
        for measure in self._list:
            measure.set_max_len(max_len)
        return self

    def set_predictor(self, prediction_model: TensorflowModelMixin, **kwargs) -> MeasureConfig:
        self.ollh.set_predictor(prediction_model)
        return self

    def set_data_distribution(self, data_distribution: DataDistribution, **kwargs) -> MeasureConfig:
        self.dllh.set_data_distribution(data_distribution)
        return self

    def init(self, **kwargs) -> MeasureConfig:
        for measure in self._list:
            measure.init(**kwargs)

        return self


# TODO: Normalise
class ViabilityMeasure(ConfigurableMixin):
    def __init__(self, vocab_len: int, max_len: int, data_distribution:DataDistribution, prediction_model: models.Model, measures: MeasureConfig = None, **kwargs) -> None:
        self.data_distribution = data_distribution
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.measures = measures
        self.measures = self.measures.set_vocab_len(vocab_len, **kwargs).set_max_len(max_len, **kwargs)
        self.measures = self.measures.set_predictor(prediction_model, **kwargs).set_data_distribution(data_distribution, **kwargs).init(**kwargs)
        self.measure_mask = MeasureMask()

    def get_config(self) -> BetterDict:
        return BetterDict(super().get_config()).merge({
            "data_distribution": len(self.data_distribution),
            "vocab_len": self.vocab_len,
            "max_len": self.max_len,
            "measure_mask": self.measure_mask.to_binstr()
        }).merge(self.measures.get_config())

    def apply_measure_mask(self, measure_mask: MeasureMask = None):
        self.measure_mask = measure_mask
        return self

    def compute(self, fa_cases: Cases, cf_cases: Cases, is_multiplied: bool = False) -> Viabilities:
        # fa_events, fa_features = fa_cases.cases
        # cf_events, cf_features = cf_cases.cases
        self.partial_values = {}
        res = Viabilities(len(cf_cases), len(fa_cases))
        result = 0 if not is_multiplied else 1
        if self.measure_mask.use_similarity:
            temp = self.measures.similarity.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_similarity(temp.T)
        if self.measure_mask.use_sparcity:
            temp = self.measures.sparcity.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_sparcity(temp.T)
        if self.measure_mask.use_dllh:
            temp = self.measures.dllh.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_dllh(temp.T)
        if self.measure_mask.use_ollh:
            temp = self.measures.ollh.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp if not is_multiplied else result * temp
            res.set_ollh(temp.T)
            res.set_mllh(self.measures.ollh.llh)
        res.set_viability(result.T)
        return res

    def __call__(self, fa_cases: Cases, cf_cases: Cases, is_multiplied=False) -> Viabilities:
        return self.compute(fa_cases, cf_cases, is_multiplied=is_multiplied)
