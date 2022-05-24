import pathlib
import tensorflow as tf
import thesis_generators.models.model_commons as commons
# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class CaseBasedGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, example_cases, topk: int = 5, *args, **kwargs):
        print(__class__)
        super(CaseBasedGeneratorModel, self).__init__(*args, **kwargs)
        self.example_cases = example_cases
        self.topk = topk


    def __call__(self, inputs):
        topk = self.topk
        fa_ev, fa_ft = inputs
        cf_ev, cf_ft = self.example_cases
        self.picks = self.compute_topk_picks(topk, fa_ev, fa_ft, cf_ev, cf_ft)
        return self.picks['events'], self.picks['features']
