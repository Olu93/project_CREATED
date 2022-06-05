import numpy as np

import thesis_commons.model_commons as commons
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.libcuts import random

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

    def predict(self, fa_case: Cases, **kwargs):
        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fa_case.cases
        _, max_len, feature_len = fa_ft.shape
        cf_ev = random.integers(0, self.vocab_len, size=(sample_size, max_len)).astype(float)
        cf_ft = random.uniform(-5, 5, size=(sample_size, max_len, feature_len))
        cf_cases = Cases(cf_ev, cf_ft)
        viab_values = self.distance.compute(fa_case, cf_cases)
        return EvaluatedCases(*cf_cases.cases, viab_values), {}
