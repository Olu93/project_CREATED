from typing import Any, Sequence, Tuple

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases, MutatedCases
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGeneratorModel
from thesis_generators.models.baselines.random_search import \
    RandomGeneratorModel
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import \
    IterationStatistics
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


    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, Any]:
        results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_ev, cf_ft, cf_viab = results.events, results.features, results.viabilities
        # cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        # if cf_viab.max() > 5:
        #     print("Something happend")
        #     cf_viab = self.evaluator(fa_case, results)
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population, info

    def construct_result(self, generation_results: Tuple[MutatedCases, Sequence[IterationStatistics]], **kwargs) -> EvaluatedCases:
        cf_results, _ = generation_results
        return cf_results


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

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, Any]:
        results, info = self.generator.predict(fa_case, sample_size=self.sample_size)
        cf_ev, cf_ft, cf_viab = results.events, results.features, results.viabilities
        # cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_viab)
        return cf_population, info

    def construct_result(self, generation_results: Tuple[MutatedCases, Sequence[IterationStatistics]], **kwargs) -> EvaluatedCases:
        cf_results, _ = generation_results
        return cf_results

