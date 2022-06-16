from typing import Any, Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorWrapper,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_commons.statististics import InstanceData, IterationData, RowData
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGenerator
from thesis_generators.models.baselines.random_search import \
    RandomGenerator
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class CaseBasedGeneratorWrapper(GeneratorWrapper):
    generator: CaseBasedGenerator = None

    def __init__(self,
                 predictor: TensorflowModelMixin,
                 generator: BaseModelMixin,
                 evaluator: ViabilityMeasure,
                 measure_mask: MeasureMask = None,
                 **kwargs) -> None:
        super().__init__(predictor, generator, evaluator,  measure_mask, **kwargs)


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, IterationData]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info, evaluated_cases=cf_population)

        return cf_population, stats
    
    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population


    def construct_instance_stats(self, info: Any, **kwargs) -> InstanceData:
        evaluated_cases: EvaluatedCases = kwargs.get('evaluated_cases')
        instance_stats: InstanceData = InstanceData()
        iter_stats: IterationData = IterationData()
        for case in evaluated_cases:
            stats_row = RowData()
            stats_row.attach('events', case.events[0])
            stats_row.attach('viability', case.viabilities.viabs[0][0])
            iter_stats.append(stats_row)
        instance_stats.append(iter_stats)
        return instance_stats



class RandomGeneratorWrapper(GeneratorWrapper):
    generator: RandomGenerator = None

    def __init__(
        self,
        predictor: TensorflowModelMixin,
        generator: BaseModelMixin,
        evaluator: ViabilityMeasure,
        measure_mask: MeasureMask = None,
        **kwargs,
    ) -> None:
        super().__init__(predictor, generator, evaluator, measure_mask, **kwargs)
        

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, InstanceData]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info, evaluated_cases=cf_population)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population

    def construct_instance_stats(self, info: Any, **kwargs) -> InstanceData:
        evaluated_cases: EvaluatedCases = kwargs.get('evaluated_cases')
        instance_stats: InstanceData = InstanceData()
        iter_stats: IterationData = IterationData()
        for case in evaluated_cases:
            stats_row = RowData()
            stats_row.attach('events', case.events[0])
            stats_row.attach('viability', case.viabilities.viabs[0][0])
            iter_stats.append(stats_row)
        instance_stats.append(iter_stats)
        return instance_stats
