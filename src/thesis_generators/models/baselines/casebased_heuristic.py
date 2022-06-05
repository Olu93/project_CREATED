import pathlib
from typing import Tuple

import tensorflow as tf
from numpy.typing import NDArray

import thesis_commons.model_commons as commons
from thesis_commons.representations import Cases, EvaluatedCases, Viabilities
from thesis_viability.viability.viability_function import ViabilityMeasure

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True

# TODO: Rename example_cases to vault
class CaseBasedGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, example_cases:Cases, evaluator: ViabilityMeasure, *args, **kwargs):
        print(__class__)
        super(CaseBasedGeneratorModel, self).__init__(name=type(self).__name__, distance=evaluator, *args, **kwargs)
        self.vault = example_cases
        self.examplars: Cases = None

    def predict(self, fc_case: Cases, **kwargs) -> EvaluatedCases:
        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fc_case.cases
        cf_cases = self.sample_vault(sample_size).examplars
        viabilities = self.distance.compute(fc_case, cf_cases)
        
        return EvaluatedCases(*cf_cases.cases, viabilities), {} # TODO: Optimize. Evaluated Cases can take from viabs
    
        
    def sample_vault(self, sample_size:int=1000):
        self.examplars = self.vault.sample(min(len(self.vault), sample_size))
        return self
    
    