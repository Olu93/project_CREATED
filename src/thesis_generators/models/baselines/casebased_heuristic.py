import pathlib
from typing import Tuple

import tensorflow as tf
from numpy.typing import NDArray

import thesis_commons.model_commons as commons
from thesis_commons.representations import Cases, EvaluatedCases
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

    def predict(self, fc_case: Cases, **kwargs):
        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fc_case.cases
        cf_ev, cf_ft = self.sample_vault(sample_size).examplars.cases
        viab_values, parts_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        return Cases(cf_ev, cf_ft, None).set_viability(viab_values), parts_values

    def compute_viabilities(self, events_input, features_input, cf_ev, cf_ft):
            viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
            partial_values = self.distance.partial_values
            return viability_values.T, partial_values
        
    def sample_vault(self, sample_size:int=1000):
        self.examplars = self.vault.sample(min(len(self.vault), sample_size))
        return self
    
    