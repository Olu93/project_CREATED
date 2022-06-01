from thesis_viability.viability.viability_function import ViabilityMeasure
import thesis_commons.model_commons as commons
import numpy as np
from thesis_commons.representations import Cases

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class RandomGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, evaluator: ViabilityMeasure, sample_size: int = 10000, *args, **kwargs):
        print(__class__)
        super(RandomGeneratorModel, self).__init__(name=type(self).__name__, distance=evaluator, *args, **kwargs)
        self.sample_size = sample_size

    def predict(self, fc_case: Cases, **kwargs):
        fa_ev, fa_ft = fc_case.data
        _, max_len, feature_len = fa_ft.shape
        cf_ev = np.random.randint(0, self.vocab_len, size=(self.sample_size, max_len)).astype(float)
        cf_ft = np.random.uniform(-5, 5, size=(self.sample_size, max_len, feature_len))
        viab_values, parts_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        return Cases(cf_ev, cf_ft, None).set_viability(viab_values), parts_values

    def compute_viabilities(self, events_input, features_input, cf_ev, cf_ft):
        viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
        partial_values = self.distance.partial_values
        return viability_values.T, partial_values