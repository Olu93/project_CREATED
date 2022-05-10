from typing import ClassVar, Generic, Type, TypeVar
import tensorflow as tf

from thesis_commons.modes import TaskModeType
from thesis_commons.libcuts import layers
import thesis_generators.models.model_commons as commons
# TODO: import thesis_commons.model_commons as commons
from thesis_commons import metric

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# TODO: Think of double stream LSTM: One for features and one for events.
# Both streams are dependendant on previous features and events.
# Requires very special loss that takes feature differences and event categorical loss into account
T = TypeVar("T", bound=commons.EmbedderLayer)


class BaseLSTM(commons.HybridInput, commons.TensorflowModelMixin):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, embed_dim=15, ff_dim=11, **kwargs):
        super(BaseLSTM, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        ft_mode = kwargs.pop('ft_mode')
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.embedder = commons.EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, mask_zero=0)
        self.lstm_layer = layers.LSTM(self.ff_dim, return_sequences=True)
        self.time_distributed_layer = layers.TimeDistributed(layers.Dense(self.vocab_len))
        self.activation_layer = layers.Activation('softmax')
        self.loss = metric.MSpCatCE()
        self.metric = [metric.MSpCatAcc(), metric.MEditSimilarity()]

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = loss or self.loss
        metrics = metrics or self.metric
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        events, features = inputs
        embeddings = self.embedder(inputs)
        y_pred = self.compute_input(embeddings)
        return y_pred

    def compute_input(self, x):
        x = self.lstm_layer(x)
        if self.time_distributed_layer is not None:
            x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred


class SimpleLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(SimpleLSTM, self).__init__(name=type(self).__name__, **kwargs)
        self.embedder = commons.TokenEmbedderLayer(self.vocab_len, self.embed_dim, mask_zero=0)

    def call(self, inputs):
        events, features = inputs
        ev_onehot = self.embedder(events)
        x = self.combiner([ev_onehot, features])
        y_pred = self.compute_input(x)
        return y_pred


class EmbeddingLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(EmbeddingLSTM, self).__init__(name=type(self).__name__, **kwargs)
        self.embedder = commons.HybridEmbedderLayer(self.vocab_len, self.embed_dim, mask_zero=0)
        del self.combiner

    def call(self, inputs):
        x = self.embedder(inputs)
        y_pred = self.compute_input(x)
        return y_pred


class OutcomeLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)

        self.activation_layer = layers.Activation('sigmoid')