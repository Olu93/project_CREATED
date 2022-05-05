import pathlib
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.lstm_cells import ProbablisticLSTMCell, ProbablisticLSTMCellV2
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils
import tensorflow as tf
from thesis_generators.models.model_commons import HybridEmbedderLayer
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import BaseModelMixin
import thesis_generators.models.model_commons as commons
from thesis_commons import metric
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType
import numpy as np
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
