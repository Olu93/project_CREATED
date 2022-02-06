from enum import Enum, auto
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_predictors.helper.metrics import MaskedEditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from thesis_commons.modes import TaskModeType, InputModeType
import inspect
import abc


class GeneratorType(Enum):
    TRADITIONAL = auto()  # Masked sparse categorical loss and metric version


class GeneratorModelMixin:
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, *args, **kwargs):
        print(__class__)
        super(GeneratorModelMixin, self).__init__(*args, **kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs


class MetricTypeMixin:

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTypeMixin, self).__init__(*args, **kwargs)


class MetricTraditionalMixin(MetricTypeMixin):

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTraditionalMixin, self).__init__(*args, **kwargs)
        self.loss = MaskedSpCatCE()
        self.metric = [MaskedSpCatAcc(), MaskedEditSimilarity()]


class CustomInputLayer(layers.Layer):
    in_layer_shape = None

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)


class TokenInputLayer(CustomInputLayer):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        print(__class__)
        super(TokenInputLayer, self).__init__(*args, **kwargs)
        self.in_layer_shape = tf.keras.layers.Input(shape=(2, max_len))

    def call(self, inputs, **kwargs):
        return self.in_layer_shape.call(inputs, **kwargs)


class HybridInputLayer(CustomInputLayer):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super(HybridInputLayer, self).__init__(*args, **kwargs)
        self.in_events = tf.keras.layers.Input(shape=(max_len, ))
        self.in_features = tf.keras.layers.Input(shape=(2, max_len, feature_len))
        self.in_layer_shape = [self.in_events, self.in_features]

    def call(self, inputs, **kwargs):
        x = [self.in_layer_shape[idx].call(inputs[idx], **kwargs) for idx in enumerate(inputs)]
        return x


class VectorInputLayer(CustomInputLayer):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super(VectorInputLayer, self).__init__(*args, **kwargs)
        self.in_layer_shape = tf.keras.layers.Input(shape=(2, max_len, feature_len))

    def call(self, inputs, **kwargs):
        return self.in_layer_shape.call(inputs, **kwargs)


class CustomEmbedderLayer(layers.Layer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super().__init__(*args, **kwargs)
        self.embedder = layers.Embedding(vocab_len, embed_dim, mask_zero=mask_zero)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)


class TokenEmbedderLayer(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(TokenEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)

    def call(self, inputs, **kwargs):
        x = self.embedder(inputs)
        return super().call(x, **kwargs)


class HybridEmbedderLayer(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(HybridEmbedderLayer, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs, **kwargs):
        indices, other_features = inputs
        embeddings = self.embedder(indices)
        features = tf.concat([embeddings, other_features], axis=-1)
        return features


class VectorEmbedderLayer(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(VectorEmbedderLayer, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs, **kwargs):
        features = inputs
        return features


class LstmInputMixin:

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(LstmInputMixin, self).__init__(*args, **kwargs)


class LSTMTokenInputMixin(LstmInputMixin):

    def __init__(self, vocab_len, max_len, feature_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(LSTMTokenInputMixin, self).__init__(vocab_len=vocab_len, max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = TokenInputLayer(max_len, feature_len)
        self.embedder = TokenEmbedderLayer(vocab_len, embed_dim, mask_zero)