from typing import Any
from thesis_commons.model_commons import BaseModelMixin, TensorflowModelMixin
from thesis_commons.representations import GeneratorResult
from thesis_commons.representations import Cases
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class SimpleVAEGenerator(GeneratorMixin):
    generator: SimpleGeneratorModel = None

    def __init__(self, predictor: TensorflowModelMixin, generator: BaseModelMixin, evaluator: ViabilityMeasure, topk:int=None, **kwargs) -> None:
        super().__init__(predictor, generator, evaluator, topk, **kwargs)

    def execute_generation(self, fc_case: Cases, **kwargs) -> Any:
        fa_events, fa_features = fc_case.data
        fa_ev_rep, fa_ft_rep = np.repeat(fa_events, 10, axis=0), np.repeat(fa_features, 10, axis=0)
        (cf_ev, cf_ft) = self.generator.predict((fa_ev_rep, fa_ft_rep))
        outcomes = self.predictor.predict((fa_events, fa_features))
        viabilities = self.evaluator(fa_events, fa_features, cf_ev, cf_ft)
        return cf_ev, cf_ft, outcomes, viabilities

    def construct_result(self, instance_num: int, generation_results: Any, **kwargs) -> GeneratorResult:
        cf_ev, cf_ft, outcomes, viabilities = generation_results
        g_result = GeneratorResult(cf_ev, cf_ft, outcomes, viabilities[0][..., None])
        return g_result

