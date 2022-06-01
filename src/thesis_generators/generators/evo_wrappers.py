from typing import Sequence, Tuple

import numpy as np

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import \
    IterationStatistics
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import \
    SimpleEvolutionStrategy
from thesis_viability.viability.viability_function import ViabilityMeasure


class SimpleEvoGeneratorWrapper(GeneratorMixin):
    generator: SimpleEvolutionStrategy = None

    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk: int = None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[MutatedCases, Sequence[IterationStatistics]]:
        fa_events, fa_features = fa_case.cases

        cf_population, stats = self.generator.predict(fa_case)
        cf_ev, cf_ft = cf_population.cases
        cf_population.set_likelihoods(self.predictor.predict((cf_ev.astype(float), cf_ft)))
        return cf_population, stats

    def construct_result(self, generation_results: Tuple[MutatedCases, Sequence[IterationStatistics]], **kwargs) -> EvaluatedCases:
        cf_population, _ = generation_results
        g_result = EvaluatedCases.from_cases(cf_population)
        return g_result
