from typing import Any, Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorWrapper, TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, EvaluatedCases
from thesis_commons.statistics import StatInstance, StatIteration, StatRow
from thesis_generators.models.baselines.baseline_search import CaseBasedGenerator, RandomGenerator, SamplingBasedGenerator
from thesis_viability.viability.viability_function import (MeasureMask, ViabilityMeasure)



class CaseBasedGeneratorWrapper(GeneratorWrapper):
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


class RandomGeneratorWrapper(GeneratorWrapper):
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

class SamplingBasedGeneratorWrapper(GeneratorWrapper):
    generator: SamplingBasedGenerator = None

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        generation_results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_population = self.construct_result(generation_results)
        stats = self.construct_instance_stats(info=info, counterfactual_cases=cf_population, factual_case=fa_case)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft, cf_viab = generation_results.events, generation_results.features, generation_results.viabilities
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population

