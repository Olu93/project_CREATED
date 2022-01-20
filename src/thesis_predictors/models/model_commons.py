import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy

from thesis_readers.helper.modes import TaskModes, TaskModeType
from ..helper.metrics import EditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from enum import IntEnum, auto, Enum
from abc import ABCMeta, abstractmethod, ABC


class ModelInterface(Model):
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, *args, **kwargs):
        super(ModelInterface, self).__init__(*args, **kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len

    def set_metrics(self):
        task_mode_type = self.task_mode_type
        assert task_mode_type is not None, f"Task mode not set. Cannot compile loss or metric. {task_mode_type if not None else 'None'} was given"
        loss_fn = None
        metric_fn = None
        if task_mode_type is TaskModeType.FIX2FIX:
            loss_fn = MaskedSpCatCE()
            metric_fn = [MaskedSpCatAcc(), EditSimilarity()]
        if task_mode_type is TaskModeType.FIX2ONE:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        if task_mode_type is TaskModeType.MANY2MANY:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        if task_mode_type is TaskModeType.MANY2ONE:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        return self

    def get_config(self):
        return self.kwargs

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        metrics = metrics or self.metric_fn
        loss = loss or self.loss_fn
        return super().compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)

    def summary(self):
        x = tf.keras.layers.Input(shape=(self.max_len, ))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class InputInterface(ABC):
    @abstractmethod
    def construct_feature_vector(self, inputs, embedder):
        raise NotImplementedError()

    @abstractmethod
    def summary(self):
        raise NotImplementedError()


class DualInput(InputInterface):
    def __init__(self, *args, **kwargs):
        super(DualInput, self).__init__()
        self.concatenate = tf.keras.layers.Concatenate()

    def construct_feature_vector(self, inputs, embedder):
        features, indices = inputs
        embeddings = embedder(indices)
        new_features = self.concat([features, embeddings])
        return new_features

    def summary(self):
        events = tf.keras.layers.Input(shape=(self.max_len, ))
        features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        inputs = [features, events]
        model = self(inputs=[inputs], outputs=self.call(inputs))
        return model.summary()


class MonoInput(InputInterface):
    def __init__(self, *args, **kwargs):
        super(MonoInput, self).__init__()

    def construct_feature_vector(self, inputs, embedder):
        indices = inputs
        embeddings = embedder(indices)
        return embeddings

    def summary(self):
        events = tf.keras.layers.Input(shape=(self.max_len, ))
        model = self(inputs=[events], outputs=self.call(events))
        return model.summary()
