import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy

from thesis_readers.helper.modes import TaskModeType, InputModeType
from ..helper.metrics import EditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from enum import IntEnum, auto, Enum
from abc import ABCMeta, abstractmethod, ABC


class InputInterface(ABC):
    @classmethod
    def summary(cls, model):
        raise NotImplementedError()


class TokenInput(InputInterface):
    input_type = InputModeType.TOKEN_INPUT

    def summary(cls, model):
        x = tf.keras.layers.Input(shape=(model.max_len, ))
        summarizer = Model(inputs=[x], outputs=model.call(x))
        return summarizer.summary()


class DualInput(InputInterface):
    input_type = InputModeType.DUAL_INPUT

    def summary(cls, model):
        events = tf.keras.layers.Input(shape=(model.max_len, ))
        features = tf.keras.layers.Input(shape=(model.max_len, model.feature_len))
        inputs = [events, features]
        summarizer = Model(inputs=[inputs], outputs=model.call(inputs))
        return summarizer.summary()


class VectorInput(InputInterface):
    input_type = InputModeType.VECTOR_INPUT

    def summary(cls, model):
        x = tf.keras.layers.Input(shape=(model.max_len, model.feature_len))
        summarizer = Model(inputs=[x], outputs=model.call(x))
        return summarizer.summary()


class ModelInterface(Model):
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    input_interface = TokenInput()
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, **kwargs):
        super(ModelInterface, self).__init__(**kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs
        self.set_metrics()

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
        config = {
            "vocab_len": self.vocab_len,
            "max_len": self.max_len,
            "feature_len": self.feature_len,
        }
        config.update(self.kwargs)
        return config

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

    def construct_feature_vector(self, inputs, embedder):
        features = None
        if self.input_interface == 0:
            indices = inputs
            features = embedder(indices)
        if self.input_interface == 1:
            indices, other_features = inputs
            embeddings = embedder(indices)
            features = tf.concat([embeddings, other_features], axis=-1)
        return features

    def summary(self):
        return self.input_interface.summary(self)
