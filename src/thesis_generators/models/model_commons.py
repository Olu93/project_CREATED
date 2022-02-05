from enum import Enum, auto
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_predictors.helper.metrics import MaskedEditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from thesis_commons.modes import TaskModeType, InputModeType


class GeneratorType(Enum):
    TRADITIONAL = auto()  # Masked sparse categorical loss and metric version


class GeneratorInterface:
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, *args, **kwargs):
        super(GeneratorInterface, self).__init__(*args, **kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs


class MetricTypeInterface:
    def __init__(self, *args, **kwargs) -> None:
        pass


class MetricTraditional(MetricTypeInterface):

    def __init__(self, *args, **kwargs) -> None:
        super(MetricTraditional, self).__init__(*args, **kwargs)
        self.loss = MaskedSpCatCE()
        self.metric = [MaskedSpCatAcc(), MaskedEditSimilarity()]

class IInputEmbedder(layers.Layer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedder = layers.Embedding(vocab_len, embed_dim, mask_zero=mask_zero)


class TokenEmbedder(IInputEmbedder):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super().__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs):
        features = self.embedder(inputs)
        return features

    # def summary(self):
    #     x = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
    #     summarizer = Model(inputs=[x], outputs=self.call(x))
    #     return summarizer.summary()

class HybridEmbedder(IInputEmbedder):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super().__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs):
        indices, other_features = inputs
        embeddings = self.embedder(indices)
        features = tf.concat([embeddings, other_features], axis=-1)
        return features


class VectorEmbedder(IInputEmbedder):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super().__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs):
        features = inputs
        return features