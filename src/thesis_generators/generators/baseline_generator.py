from typing import Any, Sequence, Tuple
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
    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator)
        self.generator: CaseBasedGeneratorModel = generator

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[Cases, Any]:
        results, info = self.generator.predict(fa_case)
        cf_ev, cf_ft, cf_viabilities = results['events'][0], results['features'][0], results['viabilities'][0][...,None]
        cf_population = Cases(cf_ev, cf_ft, None)
        cf_population.set_viability(cf_viabilities)
        return cf_population, info

    def construct_result(self, instance_num: int, generation_results: Tuple[Population, Sequence[IterationStatistics]], **kwargs) -> GeneratorResult:
        cf_population, _ = generation_results
        # cf_ev, cf_ft = cf_population.data
        cf_results = cf_population
        outcomes = self.predictor.predict(cf_results.data)
        g_result = GeneratorResult(cf_results.events, cf_results.features, outcomes, cf_results.viability_values)
        return g_result
