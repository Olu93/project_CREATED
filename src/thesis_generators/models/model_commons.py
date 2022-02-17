from enum import Enum, auto
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_commons.metrics import JoinedLoss
import thesis_commons.metrics as c_metrics
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
        super(GeneratorModelMixin, self).__init__()
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs


class MetricTypeMixin:

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricTypeMixin, self).__init__(*args, **kwargs)


class MetricVAEMixin(MetricTypeMixin):

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(MetricVAEMixin, self).__init__(*args, **kwargs)
        self.rec_loss = c_metrics.VAEReconstructionLoss()
        self.kl_loss = c_metrics.VAEKullbackLeibnerLoss()
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

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)


class TokenEmbedderLayer(EmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(TokenEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)

    def call(self, inputs, **kwargs):
        features = self.embedder(inputs[0])
        self.feature_len = features.shape[-1]
        return features


class HybridEmbedderLayer(EmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        super(HybridEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)

    def call(self, inputs, **kwargs):
        indices, other_features = inputs
        embeddings = self.embedder(indices)
        features = tf.concat([embeddings, other_features], axis=-1)
        self.feature_len = features.shape[-1]
        return features


class VectorEmbedderLayer(EmbedderLayer):

    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(VectorEmbedderLayer, self).__init__(vocab_len, embed_dim, mask_zero)

    def call(self, inputs, **kwargs):
        features = inputs[0]
        self.feature_len = features.shape[-1]
        return features


class LstmInputMixin(Model):

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


class JointTrainMixin:

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(JointTrainMixin, self).__init__(*args, **kwargs)
        self.optimizer = optimizers.Adam()

    def construct_loss(self, loss, default_losses):
        loss = (JoinedLoss(loss) if type(loss) is list else loss) if loss else (JoinedLoss(default_losses) if type(default_losses) is list else default_losses)
        return loss

    def construct_metrics(self, loss, metrics, default_metrics):
        metrics = [loss] + metrics if metrics else [loss] + default_metrics
        if type(loss) is JoinedLoss:
            metrics = loss.composites + metrics
        return metrics


class GeneratorPartMixin(GeneratorModelMixin, JointTrainMixin, Model):

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(GeneratorPartMixin, self).__init__(*args, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        default_losses = [c_metrics.VAEReconstructionLoss(name="rec"), c_metrics.VAEKullbackLeibnerLoss(name="kl")]
        default_metrics = []
        optimizer = optimizer or self.optimizer
        loss = self.construct_loss(loss, default_losses)
        metrics = self.construct_metrics(loss, metrics, default_metrics)
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)


class InterpretorPartMixin(GeneratorModelMixin, JointTrainMixin, Model):

    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(InterpretorPartMixin, self).__init__(*args, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        default_losses = c_metrics.MSpCatCE(name="cat_ce")
        default_metrics = [c_metrics.MSpCatAcc(name="cat_acc"), c_metrics.MEditSimilarity(name="ed_sim")]
        optimizer = optimizer or self.optimizer
        loss = self.construct_loss(loss, default_losses)
        metrics = self.construct_metrics(loss, metrics, default_metrics)
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)