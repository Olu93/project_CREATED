from typing import Any, Tuple

import numpy as np

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel
from thesis_viability.viability.viability_function import MeasureMask, ViabilityMeasure


class SimpleVAEGeneratorWrapper(GeneratorMixin):
    generator: SimpleGeneratorModel = None

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
        fa_events, fa_features = fa_case.cases
        fa_ev_rep, fa_ft_rep = np.repeat(fa_events, self.sample_size, axis=0), np.repeat(fa_features, self.sample_size, axis=0)
        (cf_ev, cf_ft) = self.generator.predict((fa_ev_rep, fa_ft_rep))
        cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        cf_viab = self.evaluator(fa_events, fa_features, cf_ev, cf_ft)
        cf_population = EvaluatedCases(cf_ev, cf_ft, cf_outc, cf_viab.T)
        if cf_viab.max() > 5:
            print("Something happend")
        return cf_population, {}

    def construct_result(self, generation_results: Any, **kwargs) -> EvaluatedCases:
        g_result, _ = generation_results
        return g_result

