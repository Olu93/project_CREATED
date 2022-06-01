from typing import Any, Sequence, Tuple
from thesis_generators.models.baselines.random_search import RandomGeneratorModel
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import IterationStatistics
from thesis_commons.representations import Population
from thesis_commons.model_commons import BaseModelMixin
from thesis_generators.models.baselines.casebased_heuristic import CaseBasedGeneratorModel
from thesis_commons.representations import GeneratorResult
from thesis_commons.representations import Cases
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class CaseBasedGenerator(GeneratorMixin):
    generator: CaseBasedGeneratorModel = None
    
    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk:int=None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[Cases, Any]:
        results, info = self.generator.predict(fa_case)
        cf_ev, cf_ft, cf_viab = results.events, results.features, results.viability_values
        cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        cf_population = Cases(cf_ev, cf_ft, cf_outc).set_viability(cf_viab)
        return cf_population, info

    def construct_result(self, instance_num: int, generation_results: Tuple[Population, Sequence[IterationStatistics]], **kwargs) -> GeneratorResult:
        cf_results, _ = generation_results
        g_result = GeneratorResult.from_cases(cf_results)
        return g_result

class RandomGenerator(GeneratorMixin):
    generator: RandomGeneratorModel = None
    
    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk:int=None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[Cases, Any]:
        results, info = self.generator.predict(fa_case)
        cf_ev, cf_ft, cf_viab = results.events, results.features, results.viability_values
        cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        cf_population = Cases(cf_ev, cf_ft, cf_outc).set_viability(cf_viab)
        return cf_population, info

    def construct_result(self, instance_num: int, generation_results: Tuple[Population, Sequence[IterationStatistics]], **kwargs) -> GeneratorResult:
        cf_results, _ = generation_results
        g_result = GeneratorResult.from_cases(cf_results)
        return g_result
