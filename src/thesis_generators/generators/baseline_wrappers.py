from typing import Any, Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_commons.statististics import InstanceData, IterationData, RowData
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGeneratorModel
from thesis_generators.models.baselines.random_search import \
    RandomGeneratorModel
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class CaseBasedGeneratorWrapper(GeneratorMixin):
    generator: CaseBasedGeneratorModel = None

    def __init__(self,
                 predictor: TensorflowModelMixin,
                 generator: BaseModelMixin,
                 evaluator: ViabilityMeasure,
                 topk: int = None,
                 measure_mask: MeasureMask = None,
                 **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, measure_mask, **kwargs)
        self.sample_size = kwargs.get('sample_size', 1000)


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, IterationData]:
        generation_results, stats = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population


class RandomGeneratorWrapper(GeneratorMixin):
    generator: RandomGeneratorModel = None

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

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, InstanceData]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_stats(info=info, evaluated_cases=cf_population)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population

    def construct_stats(self, info: Any, **kwargs) -> InstanceData:
        evaluated_cases: EvaluatedCases = kwargs.get('evaluated_cases')
        instance_stats: InstanceData = InstanceData()
        
        
        iter_stats: IterationData = IterationData()

        for case in evaluated_cases:
            stats_row = RowData()
            stats_row.attach('events', case.events)
            stats_row.attach('viability', case.viab)
            iter_stats.append(stats_row)
        instance_stats.append(iter_stats)
        return instance_stats
