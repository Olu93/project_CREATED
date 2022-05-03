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
        events_input, features_input = inputs
        batch_size, sequence_length, feature_len = features_input.shape
        cf_ev, cf_ft = self.example_cases
        viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
        best_values_indices = np.argsort(viability_values, axis=1)
        chosen = np.where((best_values_indices >= (len(cf_ev) - self.topk)))
        chosen_ft_shape = (batch_size, self.topk, self.max_len, -1)
        chosen_ev_shape = chosen_ft_shape[:3]
        chosen_viab_shape = chosen_ft_shape[:2]
        chosen_ev_flattened, chosen_ft_flattened = cf_ev[chosen[1]], cf_ft[chosen[1]]
        chosen_ev, chosen_ft = chosen_ev_flattened.reshape(chosen_ev_shape), chosen_ft_flattened.reshape(chosen_ft_shape)
        self.chosen_viabilities = viability_values[chosen[0], chosen[1]].reshape(chosen_viab_shape)
        return chosen_ev, chosen_ft


