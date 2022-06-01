from typing import Any, Tuple

import numpy as np

from thesis_commons.model_commons import (BaseModelMixin, GeneratorMixin,
                                          TensorflowModelMixin)
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel
from thesis_viability.viability.viability_function import ViabilityMeasure


class SimpleVAEGeneratorWrapper(GeneratorMixin):
    generator: SimpleGeneratorModel = None

    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk:int=None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)
        self.sample_size = kwargs.get('sample_size', 1000)

    def execute_generation(self, fa_case: Cases, **kwargs) -> Tuple[Cases, Any]:
        fa_events, fa_features = fa_case.cases
        fa_ev_rep, fa_ft_rep = np.repeat(fa_events, self.sample_size, axis=0), np.repeat(fa_features, self.sample_size, axis=0)
        (cf_ev, cf_ft) = self.generator.predict((fa_ev_rep, fa_ft_rep))
        cf_outc = self.predictor.predict((cf_ev.astype(float), cf_ft))
        cf_viab = self.evaluator(fa_events, fa_features, cf_ev, cf_ft)
        cf_population = Cases(cf_ev, cf_ft, cf_outc).set_viability(cf_viab[0][..., None])
        return cf_population, {}

    def construct_result(self, generation_results: Any, **kwargs) -> EvaluatedCases:
        cf_results, _ = generation_results
        g_result = EvaluatedCases.from_cases(cf_results)
        return g_result

