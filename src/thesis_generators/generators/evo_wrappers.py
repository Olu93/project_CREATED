from typing import Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases, MutationRate
from thesis_commons.statististics import InstanceData, IterationData
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy
# from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import \
#     SimpleEvolutionStrategy
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class SimpleEvoGeneratorWrapper(GeneratorMixin):
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


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[MutatedCases, InstanceData]:
        # fa_events, fa_features = fa_case.cases

        generation_results, info = self.generator.predict(fa_case)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info)
        return cf_population, stats

    def construct_result(self, cf_population: MutatedCases, **kwargs) -> EvaluatedCases:
        return cf_population

    def construct_instance_stats(self, info, **kwargs) -> IterationData:
        return info
    
    def construct_model_stats(self, **kwargs) -> None:
        super().construct_model_stats(**kwargs)
        self.run_stats.attach('hparams.mrate', self.generator.mutation_rate.to_dict())