import numpy as np

import thesis_commons.model_commons as commons
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_viability.viability.viability_function import ViabilityMeasure

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class RandomGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, evaluator: ViabilityMeasure, *args, **kwargs):
        print(__class__)
        super(RandomGeneratorModel, self).__init__(name=type(self).__name__, distance=evaluator, *args, **kwargs)

    def predict(self, fc_case: Cases, **kwargs):
        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fc_case.cases
        _, max_len, feature_len = fa_ft.shape
        cf_ev = np.random.randint(0, self.vocab_len, size=(sample_size, max_len)).astype(float)
        cf_ft = np.random.uniform(-5, 5, size=(sample_size, max_len, feature_len))
        viab_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        return EvaluatedCases(cf_ev, cf_ft, viab_values.mllh, viab_values), {}

    def compute_viabilities(self, events_input, features_input, cf_ev, cf_ft): # TODO: Not necessary anymore
        viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
        return viability_values