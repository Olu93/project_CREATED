import pathlib
from typing import Sequence, Tuple
from thesis_commons.constants import PATH_RESULTS_MODELS_SPECIFIC

from thesis_commons.model_commons import (BaseModelMixin, GeneratorWrapper,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases, MutationRate
from thesis_commons.statistics import StatInstance, StatIteration
from thesis_generators.models.evolutionary_strategies.evolutionary_strategy import EvolutionaryStrategy
# from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import \
#     SimpleEvolutionStrategy
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class EvoGeneratorWrapper(GeneratorWrapper):
    generator: EvolutionaryStrategy = None

    def __init__(
        self,
        predictor: TensorflowModelMixin,
        generator: BaseModelMixin,
        evaluator: ViabilityMeasure,
        topk: int = None,
        measure_mask: MeasureMask = None,
        **kwargs,
    ) -> None:
        super().__init__(predictor, generator, evaluator, measure_mask, **kwargs)


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        generation_results, info = self.generator.predict(fa_case)
        cf_population = self.construct_result(generation_results)
        stats = info
        return cf_population, stats

    def construct_result(self, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        return cf_population

    
