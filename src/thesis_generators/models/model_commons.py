from enum import Enum, auto
import tensorflow as tf
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics
# from tensorflow.keras import Model, layers, optimizers
# from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_commons import metric
from thesis_commons.modes import TaskModeType, InputModeType
import inspect
import abc


# TODO: Fix imports by collecting all commons
# TODO: Rename to 'SamplingLayer'
class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        # Why log(x) - https://stats.stackexchange.com/a/486161
        z_mean, z_log_var = inputs
        # Why log(variance) - https://stats.stackexchange.com/a/486205

        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ReverseEmbedding(layers.Layer):
    def __init__(self, embedding_layer: layers.Embedding, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic)
        self.embedding_layer = embedding_layer

    def call(self, inputs, **kwargs):
        B = self.embedding_layer.get_weights()[0]
        A = K.reshape(inputs, (-1, B.shape[1]))
        similarities = self.cosine_similarity_faf(A, B)
        indices = K.argmax(similarities)
        indices_reshaped = tf.reshape(indices, inputs.shape[:2])
        indices_onehot = tf.keras.utils.to_categorical(indices_reshaped, A.shape[1])

        return indices_onehot

    def cosine_similarity_faf(self, A, B):
        nominator = A @ B
        norm_A = tf.norm(A, axis=1)
        norm_B = tf.norm(B, axis=1)
        denominator = tf.reshape(norm_A, [-1, 1]) * tf.reshape(norm_B, [1, -1])
        return tf.divide(nominator, denominator)


class GeneratorType(Enum):
    TRADITIONAL = auto()  # Masked sparse categorical loss and metric version


class BaseModelMixin:
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    loss_fn: losses.Loss = None
    metric_fn: metrics.Metric = None

    def __init__(self, vocab_len, max_len, feature_len, *args, **kwargs):
        print(__class__)
        super(BaseModelMixin, self).__init__()
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs


class JointTrainMixin:
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(JointTrainMixin, self).__init__(*args, **kwargs)
        self.optimizer = optimizers.Adam()

    def construct_loss(self, loss, default_losses):
        loss = (metric.JoinedLoss(loss) if type(loss) is list else loss) if loss else (metric.JoinedLoss(default_losses) if type(default_losses) is list else default_losses)
        return loss

    def construct_metrics(self, loss, metrics, default_metrics):
        metrics = [loss] + metrics if metrics else [loss] + default_metrics
        if type(loss) is metric.JoinedLoss:
            metrics = loss.composites + metrics
        return metrics

class HybridGraph():
    def __init__(self, *args, **kwargs) -> None:
        super(HybridGraph, self).__init__(*args, **kwargs)
        self.in_events = tf.keras.layers.Input(shape=(self.max_len, ))  # TODO: Fix import
        self.in_features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        self.in_layer_shape = [self.in_events, self.in_features]

    def build_graph(self):
        events = layers.Input(shape=(self.max_len, ))
        features = layers.Input(shape=(self.max_len, self.feature_len))
        inputs = [events, features]
        summarizer = models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer

class TensorflowModelMixin(BaseModelMixin, JointTrainMixin, models.Model):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(TensorflowModelMixin, self).__init__(*args, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        optimizer = optimizer or self.optimizer
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

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

    def build_graph(self):
        events = tf.keras.layers.Input(shape=(self.max_len, ), name="events")
        features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len), name="event_attributes")
        inputs = [events, features]
        summarizer = models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer

class InterpretorPartMixin(BaseModelMixin, JointTrainMixin, models.Model):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(InterpretorPartMixin, self).__init__(*args, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        optimizer = optimizer or self.optimizer
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)


class MetricTypeMixin:
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTypeMixin, self).__init__(*args, **kwargs)


class MetricVAEMixin(MetricTypeMixin):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricVAEMixin, self).__init__(*args, **kwargs)
        self.rec_loss = metric.GaussianReconstructionLoss()
        self.kl_loss = metric.SimpleKLDivergence()
        self.loss = None
        self.metric = None

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, z_mean: tf.Tensor, z_log_var: tf.Tensor):
        rec_loss = self.rec_loss(y_true, y_pred) * y_true.shape[1]
        kl_loss = self.kl_loss(z_mean, z_log_var) * y_true.shape[1]
        return {
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
        }


class InputModeTypeDetector:
    pass  # Maybe override build


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
        self.in_layer_shape = tf.keras.layers.Input(shape=(max_len, ))

    def call(self, inputs, **kwargs):
        return self.in_layer_shape.call(inputs, **kwargs)




class HybridInputLayer(CustomInputLayer):
    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super(HybridInputLayer, self).__init__(*args, **kwargs)
        self.in_events = tf.keras.layers.Input(shape=(max_len, ))  # TODO: Fix import
        self.in_features = tf.keras.layers.Input(shape=(max_len, feature_len))
        self.in_layer_shape = [self.in_events, self.in_features]

    def call(self, inputs, **kwargs):
        x = [self.in_layer_shape[idx].call(inputs[idx], **kwargs) for idx in enumerate(inputs)]
        return x


class VectorInputLayer(CustomInputLayer):
    def __init__(self, max_len, feature_len, *args, **kwargs) -> None:
        super(VectorInputLayer, self).__init__(*args, **kwargs)
        self.in_layer_shape = tf.keras.layers.Input(shape=(max_len, feature_len))

    def call(self, inputs, **kwargs):
        return self.in_layer_shape.call(inputs, **kwargs)


class EmbedderLayer(layers.Layer):
    def __init__(self, feature_len=None, max_len=None, ff_dim=None, vocab_len=None, embed_dim=None, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(EmbedderLayer, self).__init__(*args, **kwargs)
        self.embedder = layers.Embedding(vocab_len, embed_dim, mask_zero=mask_zero, *args, **kwargs)
        self.feature_len: int = None
        # self.vocab_len: int = vocab_len

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)


class OnehotEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(OnehotEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)
        # self.embedder = layers.CategoryEncoding(vocab_len, output_mode="one_hot")
        # self.test = layers.Lambda(lambda ev_sequence: self.embedder(ev_sequence))
        self.embedder = layers.Lambda(self._one_hot, arguments={'num_classes': vocab_len})

    # Helper method (not inlined for clarity)
    def _one_hot(self, x, num_classes):
        return K.one_hot(K.cast(x, tf.uint8), num_classes=num_classes)

    def call(self, inputs, **kwargs):
        indices = inputs
        # features = self.test(indices)
        features = self.embedder(indices)
        self.feature_len = features.shape[-1]
        return features


class OneHotEncodingLayer():
    # https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
    def __init__(self, input_dim=None, input_length=None) -> None:
        # Check if inputs were supplied correctly
        if input_dim is None or input_length is None:
            raise TypeError("input_dim or input_length is not set")
        self.input_dim = input_dim
        self.input_length = input_length
        self.embedder = layers.Lambda(self._one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length, ))

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, tf.uint16), num_classes=num_classes)

    def call(self, input):
        # Final layer representation as a Lambda layer
        return self.embedder(input)


class TokenEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(TokenEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)

    def call(self, inputs, **kwargs):
        indices = inputs
        features = self.embedder(indices)
        self.feature_len = features.shape[-1]
        return features


class HybridEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        super(HybridEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)
        self.concatenator = layers.Concatenate()

    def call(self, inputs, **kwargs):
        indices, other_features = inputs
        embeddings = self.embedder(indices)
        features = self.concatenator([embeddings, other_features])
        self.feature_len = features.shape[-1]
        return features


class VectorEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(VectorEmbedderLayer, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs, **kwargs):
        features = inputs[0]
        self.feature_len = features.shape[-1]
        return features


class LstmInputMixin(models.Model):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(LstmInputMixin, self).__init__(*args, **kwargs)


class LSTMTokenInputMixin(LstmInputMixin):
    def __init__(self, vocab_len, max_len, feature_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(LSTMTokenInputMixin, self).__init__(vocab_len=vocab_len, max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = TokenInputLayer(max_len, feature_len)
        self.embedder = TokenEmbedderLayer(vocab_len, embed_dim, mask_zero)


class LSTMVectorInputMixin(LstmInputMixin):
    def __init__(self, vocab_len, max_len, feature_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(LSTMVectorInputMixin, self).__init__(vocab_len=vocab_len, max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = VectorInputLayer(max_len, feature_len)
        self.embedder = VectorEmbedderLayer(vocab_len, embed_dim, mask_zero)


class LSTMHybridInputMixin(LstmInputMixin):
    def __init__(self, vocab_len, max_len, feature_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(LSTMHybridInputMixin, self).__init__(vocab_len=vocab_len, max_len=max_len, feature_len=feature_len, *args, **kwargs)
        self.in_layer = HybridInputLayer(max_len, feature_len)
        self.embedder = HybridEmbedderLayer(vocab_len, embed_dim, mask_zero)


class InputInterface(abc.ABC):
    @classmethod
    def summary(self):
        raise NotImplementedError()


class TokenInput(InputInterface):
    input_type = InputModeType.TOKEN_INPUT

    def summary(self):
        x = tf.keras.layers.Input(shape=(self.max_len, ))
        summarizer = models.Model(inputs=[x], outputs=self.call(x))
        return summarizer.summary()


class HybridInput(InputInterface):
    input_type = InputModeType.DUAL_INPUT

    def summary(self):
        events = tf.keras.layers.Input(shape=(self.max_len, ))
        features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        inputs = [events, features]
        summarizer = models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary()


class VectorInput(InputInterface):
    input_type = InputModeType.VECTOR_INPUT

    def summary(self):
        x = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        summarizer = models.Model(inputs=[x], outputs=self.call(x))
        return summarizer.summary()
