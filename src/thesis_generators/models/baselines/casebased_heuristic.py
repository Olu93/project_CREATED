import pathlib
from typing import Tuple
import tensorflow as tf
from thesis_commons.representations import GeneratorResult
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.representations import Cases
import thesis_commons.model_commons as commons
from numpy.typing import NDArray
# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class CaseBasedGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, example_cases, evaluator: ViabilityMeasure, *args, **kwargs):
        print(__class__)
        super(CaseBasedGeneratorModel, self).__init__(name=type(self).__name__, distance=evaluator, *args, **kwargs)
        self.example_cases = example_cases

    def predict(self, fc_case: Cases, **kwargs):
        fa_ev, fa_ft = fc_case.data
        cf_ev, cf_ft = self.example_cases
        viab_values, parts_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        return Cases(cf_ev, cf_ft, None).set_viability(viab_values), parts_values

    def compute_viabilities(self, events_input, features_input, cf_ev, cf_ft):
            viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
            partial_values = self.distance.partial_values
            return viability_values.T, partial_values
    