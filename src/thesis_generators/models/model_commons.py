from enum import Enum, auto
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_predictors.helper.metrics import MaskedEditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from thesis_commons.modes import TaskModeType, InputModeType
import inspect


class GeneratorType(Enum):
    TRADITIONAL = auto()  # Masked sparse categorical loss and metric version


class GeneratorInterface:
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, *args, **kwargs):
        print(__class__)
        super(GeneratorInterface, self).__init__(*args, **kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs


class MetricTypeInterface:

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTypeInterface, self).__init__(*args, **kwargs)


class MetricTraditional(MetricTypeInterface):

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTraditional, self).__init__(*args, **kwargs)
        self.loss = MaskedSpCatCE()
        self.metric = [MaskedSpCatAcc(), MaskedEditSimilarity()]


class InputLayerMixin:
    in_layer: layers.Input = None

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(InputLayerMixin, self).__init__(*args, **kwargs)

    def summary(self) -> None:
        x = self.in_layer
        summarizer = Model(inputs=[x], outputs=self.call(x))
        return summarizer.summary()

    def return_in_layer(self) -> layers.Input:
        return self.in_layer


class TokenInputMixin(InputLayerMixin):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        print(__class__)
        super().__init__(max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = tf.keras.layers.Input(shape=(max_len, ))


class HybridInputMixin(InputLayerMixin):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super().__init__(max_len=max_len, feature_len=feature_len, *args, **kwargs)
        events = tf.keras.layers.Input(shape=(self.max_len, ))
        features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        self.in_layer = [events, features]


class VectorInputMixin(InputLayerMixin):

    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super().__init__(max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = tf.keras.layers.Input(shape=(max_len, feature_len))


class CustomEmbedderLayer(layers.Layer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(CustomEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, *args, **kwargs)
        self.embedder = layers.Embedding(vocab_len, embed_dim, mask_zero=mask_zero)


class TokenEmbedder(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(TokenEmbedder, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)

    def call(self, inputs):
        x = self.embedder(inputs)
        return x


class HybridEmbedder(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(HybridEmbedder, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs):
        indices, other_features = inputs
        embeddings = self.embedder(indices)
        features = tf.concat([embeddings, other_features], axis=-1)
        return features


class VectorEmbedder(CustomEmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(VectorEmbedder, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs):
        features = inputs
        return features