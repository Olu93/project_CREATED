from typing import Any, Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorWrapper, TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_commons.statististics import StatInstance, StatIteration, StatRow
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGenerator
from thesis_generators.models.baselines.random_search import \
    RandomGenerator
from thesis_viability.viability.viability_function import (MeasureMask, ViabilityMeasure)


class BaselineWrapperInterface(GeneratorWrapper):
    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, measure_mask: MeasureMask = None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, measure_mask, **kwargs)

    def construct_instance_stats(self, info: Any, **kwargs) -> StatInstance:
        counterfactual_cases: EvaluatedCases = kwargs.get('counterfactual_cases')
        factual_case: EvaluatedCases = kwargs.get('factual_case')
        instance_stats: StatInstance = StatInstance()
        iter_stats: StatIteration = StatIteration()
        case: EvaluatedCases = None
        for case in counterfactual_cases:
            stats_row = StatRow()
            stats_row.attach('factual_outcome', factual_case.outcomes[0][0])
            stats_row.attach('target_outcome', ~factual_case.outcomes[0][0])
            stats_row.attach('predicted_outcome', case.outcomes[0][0])
            stats_row.attach('prediction_score', case.viabilities.mllh[0][0])
            stats_row.attach('similarity', case.viabilities.similarity[0][0])
            stats_row.attach('sparcity', case.viabilities.sparcity[0][0])
            stats_row.attach('dllh', case.viabilities.dllh[0][0])
            stats_row.attach('delta', case.viabilities.ollh[0][0])
            stats_row.attach('viability', case.viabilities.viabs[0][0])
            stats_row.attach('events', case.events[0])
            iter_stats.append(stats_row)

        iter_stats.attach(f"n_results", len(counterfactual_cases))
        iter_stats.attach(f"avg_viability", counterfactual_cases.avg_viability[0])
        iter_stats.attach(f"avg_viability", counterfactual_cases.avg_viability[0])
        iter_stats.attach(f"median_viability", counterfactual_cases.median_viability[0])
        iter_stats.attach(f"max_viability", counterfactual_cases.max_viability[0])
        iter_stats.attach(f"min_viability", counterfactual_cases.min_viability[0])

        instance_stats.append(iter_stats)
        return instance_stats


class CaseBasedGeneratorWrapper(BaselineWrapperInterface):
    generator: CaseBasedGenerator = None

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatIteration]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info, counterfactual_cases=cf_population, factual_case=fa_case)

        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population


class RandomGeneratorWrapper(BaselineWrapperInterface):
    generator: RandomGenerator = None

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info, counterfactual_cases=cf_population, factual_case=fa_case)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population
