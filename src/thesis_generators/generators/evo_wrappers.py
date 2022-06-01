from typing import Sequence, Tuple
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import IterationStatistics
from thesis_commons.representations import Population
from thesis_commons.model_commons import BaseModelMixin
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import SimpleEvolutionStrategy
from thesis_commons.representations import GeneratorResult
from thesis_commons.representations import Cases
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class SimpleEvoGeneratorWrapper(GeneratorMixin):
    generator: SimpleEvolutionStrategy = None

    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk: int = None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[Population, Sequence[IterationStatistics]]:
        fa_events, fa_features = fa_case.data

        cf_population, stats = self.generator.predict(fa_case)
        cf_ev, cf_ft = cf_population.data
        cf_population.outcomes = self.predictor.predict((cf_ev.astype(float), cf_ft))
        return cf_population, stats

    def construct_result(self, generation_results: Tuple[Population, Sequence[IterationStatistics]], **kwargs) -> GeneratorResult:
        cf_population, _ = generation_results
        g_result = GeneratorResult.from_cases(cf_population)
        return g_result
