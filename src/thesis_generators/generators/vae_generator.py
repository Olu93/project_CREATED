from typing import Any
from src.thesis_commons.model_commons import BaseModelMixin
from thesis_commons.representations import GeneratorResult
from thesis_commons.representations import Cases
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class VAEGenerator(GeneratorMixin):
    def __init__(self, generator: BaseModelMixin, evaluator: ViabilityMeasure, **kwargs) -> None:
        super().__init__(evaluator)
        self.generator: SimpleGeneratorModel = generator

    def execute_generation(self, fc_case, **kwargs) -> Any:
        fa_events, fa_features = fc_case.items()
        (cf_ev, cf_ft) = self.generator.predict([np.repeat(fa_events, 10, axis=0), np.repeat(fa_features, 10, axis=0)])
        outcomes = self.generator.predict((fa_events, fa_features))
        viabilities = self.evaluator(fa_events, fa_features, cf_ev, cf_ft)
        return cf_ev, cf_ft, outcomes, viabilities
    
    def construct_result(self, instance_num:int, generation_results:Any, **kwargs) -> GeneratorResult:
        cf_ev, cf_ft, outcomes, vals = generation_results
        g_result = GeneratorResult(cf_ev, cf_ft, outcomes, vals)
        return g_result
            
        
