from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Sequence
from thesis_commons.distributions import DataDistribution
if TYPE_CHECKING:
    from thesis_commons.model_commons import TensorflowModelMixin

from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, ConfigurationSet, Viabilities
import pandas as pd
import itertools as it

import numpy as np
import tensorflow as tf

keras = tf.keras
from keras import backend as K, losses, metrics, utils, layers, optimizers, models
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


# TODO: Incoporate - inter list diversity
class ViabilityMeasure(ConfigurableMixin):
    def __init__(self, vocab_len: int, max_len: int, data_distribution: DataDistribution, prediction_model: models.Model, measures: MeasureConfig = None, **kwargs) -> None:
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

    def compute(self, fa_cases: Cases, cf_cases: Cases) -> Viabilities:
        # fa_events, fa_features = fa_cases.cases
        # cf_events, cf_features = cf_cases.cases
        self.partial_values = {}
        res = Viabilities(len(cf_cases), len(fa_cases))
        result = 0
        if self.measure_mask.use_similarity:
            temp = self.measures.similarity.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp
            res.set_similarity(temp.T)
        if self.measure_mask.use_sparcity:
            temp = self.measures.sparcity.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp
            res.set_sparcity(temp.T)
        if self.measure_mask.use_dllh:
            temp = self.measures.dllh.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp
            res.set_dllh(temp.T)
        if self.measure_mask.use_ollh:
            temp = self.measures.ollh.compute_valuation(fa_cases, cf_cases).normalize().normalized_results
            result = result + temp
            res.set_ollh(temp.T)
            res.set_mllh(self.measures.ollh.llh)
        res.set_viability(result.T)
        return res

    def __call__(self, fa_cases: Cases, cf_cases: Cases) -> Viabilities:
        return self.compute(fa_cases, cf_cases)


class ParetoRankedViabilityMeasure(ViabilityMeasure):
    def compute(self, fa_cases: Cases, cf_cases: Cases) -> Viabilities:
        # fa_events, fa_features = fa_cases.cases
        # cf_events, cf_features = cf_cases.cases
        self.partial_values = {}
        res = Viabilities(len(cf_cases), len(fa_cases))
        result = 0
        if self.measure_mask.use_similarity:
            temp = self.measures.similarity.compute_valuation(fa_cases, cf_cases)
            res.set_similarity(temp.T)
        if self.measure_mask.use_sparcity:
            temp = self.measures.sparcity.compute_valuation(fa_cases, cf_cases)
            res.set_sparcity(temp.T)
        if self.measure_mask.use_dllh:
            temp = self.measures.dllh.compute_valuation(fa_cases, cf_cases)
            res.set_dllh(temp.T)
        if self.measure_mask.use_ollh:
            temp = self.measures.ollh.compute_valuation(fa_cases, cf_cases)
            res.set_ollh(temp.T)
            res.set_mllh(self.measures.ollh.llh)

        # tmp_res = res
        # tmp_cases = cf_cases
        # round_1 = np.stack(tmp_res.get_parts(Viabilities.Measures.OUTPUT_LLH), tmp_res.get_parts(Viabilities.Measures.DATA_LLH)).T
        # mask = self.is_pareto_efficient(round_1)
        # nondominated_1 = np.where(tmp_cases[mask])[0]
        # dominated = np.where(tmp_cases[~mask])[0]

        # tmp_res = tmp_res[dominated]
        # tmp_cases = tmp_cases[dominated]
        # round_2 = np.stack(tmp_res.get_parts(Viabilities.Measures.DATA_LLH), tmp_res.get_parts(Viabilities.Measures.SPARCITY)).T
        # mask = self.is_pareto_efficient(round_2)
        # nondominated_2 = np.where(tmp_cases[mask])[0]
        # dominated = np.where(tmp_cases[~mask])[0]

        # tmp_res = tmp_res[dominated]
        # tmp_cases = tmp_cases[dominated]
        # round_3 = np.stack(tmp_res.get_parts(Viabilities.Measures.SPARCITY), tmp_res.get_parts(Viabilities.Measures.SIMILARITY)).T
        # mask = self.is_pareto_efficient(round_3)
        # nondominated_3 = np.where(tmp_cases[mask])[0]
        # dominated = np.where(tmp_cases[~mask])[0]

        # non_dominated_4 = tmp_cases[dominated]
        M = Viabilities.Measures
        importance = [M.OUTPUT_LLH, M.DATA_LLH, M.SPARCITY, M.SIMILARITY, M.MODEL_LLH]
        parts = res._parts[importance]
        tmp = pd.DataFrame(parts)
        tmp2 = tmp.sort_values([0,1,2,3], ascending=False)
        res = Viabilities(len(cf_cases), len(fa_cases))
        res = res.set_ollh(tmp2.iloc[:, importance.index(M.OUTPUT_LLH)])
        res = res.set_dllh(tmp2.iloc[:, importance.index(M.DATA_LLH)])
        res = res.set_sparcity(tmp2.iloc[:, importance.index(M.SPARCITY)])
        res = res.set_similarity(tmp2.iloc[:, importance.index(M.SIMILARITY)])
        res = res.set_ollh(tmp2.iloc[:, importance.index(M.MODEL_LLH)])
        new_viab = ((len(tmp2)-np.arange(0, len(tmp2)))/len(tmp2))[:, len(fa_cases)]
        res = res.set_viability(new_viab)
        
        
        return res

    def is_pareto_efficient_dumb(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
        return is_efficient

    # Fairly fast for many datapoints, less fast for many costs, somewhat readable
    def is_pareto_efficient_simple(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    # Faster than is_pareto_efficient_simple, but less readable.
    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient