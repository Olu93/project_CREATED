from typing import Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_commons.statististics import InstanceData, IterationData
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import \
    SimpleEvolutionStrategy
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class SimpleEvoGeneratorWrapper(GeneratorMixin):
    generator: SimpleEvolutionStrategy = None

    def __init__(
        self,
        predictor: TensorflowModelMixin,
        generator: BaseModelMixin,
        evaluator: ViabilityMeasure,
        topk: int = None,
        measure_mask: MeasureMask = None,
        **kwargs,
    ) -> None:
        super().__init__(predictor, generator, evaluator, topk, measure_mask, **kwargs)
        self.sample_size = kwargs.get('sample_size', 1000)


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[MutatedCases, Sequence[InstanceData]]:
        # fa_events, fa_features = fa_case.cases

        generation_results, stats = self.generator.predict(fa_case)
        cf_population = self.construct_result(generation_results)
        return cf_population, stats

    def construct_result(self, cf_population: MutatedCases, **kwargs) -> EvaluatedCases:
        return cf_population
