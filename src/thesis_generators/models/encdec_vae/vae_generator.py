from src.thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class VAEGenerator(GeneratorMixin):
    def __init__(self, evaluator: ViabilityMeasure, **kwargs) -> None:
        super().__init__(evaluator)
        self.generator: SimpleGeneratorModel = kwargs.pop('generator')
        
    def generate(self, fa_seeds, fa_labels):
        fa_events, fa_features = fa_seeds
        return self.generator.predict([np.repeat(fa_events, 10, axis=0),np.repeat(fa_features, 10, axis=0) ])   