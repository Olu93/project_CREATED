import pathlib
import tensorflow as tf
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.representations import Cases
import thesis_commons.model_commons as commons
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
        topk = kwargs.pop('topk', 5)
        fa_ev, fa_ft = fc_case.data
        cf_ev, cf_ft = self.example_cases
        picks = self.compute_topk_picks(topk, fa_ev, fa_ft, cf_ev, cf_ft)
        return picks
