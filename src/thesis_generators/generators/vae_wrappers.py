from typing import Any, Sequence, Tuple

import numpy as np

from thesis_commons.model_commons import (BaseModelMixin, GeneratorWrapper,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_commons.statistics import StatInstance, StatIteration, StatRow
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)


class SimpleVAEGeneratorWrapper(GeneratorWrapper):
    generator: SimpleLSTMGeneratorModel = None

    def __init__(
        self,
        predictor: TensorflowModelMixin,
        generator: BaseModelMixin,
        evaluator: ViabilityMeasure,
        measure_mask: MeasureMask = None,
        **kwargs,
    ) -> None:
        super().__init__(predictor, generator, evaluator, measure_mask, **kwargs)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        fa_events, fa_features = fa_case.cases
        fa_ev_rep, fa_ft_rep = np.repeat(fa_events, self.sample_size, axis=0), np.repeat(fa_features, self.sample_size, axis=0)
        generation_results = self.generator.predict((fa_ev_rep, fa_ft_rep))
        cf_population = self.construct_result(Cases(*generation_results), fa_case=fa_case)
        stats = self.construct_instance_stats(info={}, counterfactual_cases=cf_population, factual_case=fa_case)
        return cf_population, stats

    def construct_result(self, generation_results: Cases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft = generation_results.cases
        cf_cases = Cases(cf_ev.astype(float), cf_ft)
        cf_viab = self.evaluator(kwargs.pop('fa_case'), cf_cases)
        cf_population = EvaluatedCases(*cf_cases.cases, cf_viab)
        return cf_population



