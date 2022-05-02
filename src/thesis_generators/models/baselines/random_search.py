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


class CaseBasedGeneratorModel(commons.TensorflowModelMixin):
    def __init__(self, examples, distance: ViabilityMeasure, sample_size: int = 10000, topk:int=5, *args, **kwargs):
        print(__class__)
        super(CaseBasedGeneratorModel, self).__init__(*args, **kwargs)
        self.examples = examples
        self.distance = distance
        self.sample_size = sample_size
        self.topk = topk

    def set_distance(self, distance: ViabilityMeasure):
        self.distance = distance

    def predict(self, inputs):
        events_input, features_input = inputs
        batch_size, sequence_length, feature_len = features_input.shape
        cf_ev = np.random.randint(0, self.vocab_len, size=(self.sample_size, self.max_len))
        cf_ft = np.random.uniform(-5, 5, size=(self.sample_size, self.max_len, self.feature_len))
        
        viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
        best_values_indices = np.argsort(viability_values, axis=0)
        shape = (len(events_input), self.topk)
        topk_per_input = best_values_indices[:, :self.topk]
        topk_per_input_flattened = topk_per_input.flatten()
        chosen_ev_flattened, chosen_ft_flattened = cf_ev[topk_per_input_flattened], cf_ft[topk_per_input_flattened]
        chosen_ev, chosen_ft = chosen_ev_flattened.reshape(shape), chosen_ft_flattened.reshape(shape + (-1, ))
        return chosen_ev, chosen_ft