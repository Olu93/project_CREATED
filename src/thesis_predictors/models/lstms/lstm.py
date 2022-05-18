from typing import ClassVar, Generic, Type, TypeVar
import tensorflow as tf
from thesis_commons.constants import REDUCTION
from thesis_commons.modes import TaskModeType
from thesis_commons.libcuts import layers, K, losses, keras
import thesis_generators.models.model_commons as commons
# TODO: import thesis_commons.model_commons as commons
from thesis_commons import metric

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
DEBUG_SHOW_ALL_METRICS = False
# TODO: Think of double stream LSTM: One for features and one for events.
# Both streams are dependendant on previous features and events.
# Requires very special loss that takes feature differences and event categorical loss into account
T = TypeVar("T", bound=commons.EmbedderLayer)


class BaseLSTM(commons.HybridInput, commons.TensorflowModelMixin):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, embed_dim=10, ff_dim=5, **kwargs):
        super(BaseLSTM, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        ft_mode = kwargs.pop('ft_mode')
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.embedder = commons.EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, mask_zero=0)
        self.lstm_layer = layers.LSTM(self.ff_dim, return_sequences=True)
        self.logit_layer = layers.TimeDistributed(layers.Dense(self.vocab_len))
        self.activation_layer = layers.Activation('softmax')
        self.custom_loss, self.custom_eval = self.init_metrics()
        # self.c = []

    def train_step(self, data):
        if len(data) == 3:
            (events_input, features_input), events_target, class_weight = data
        else:
            sample_weight = None
            (events_input, features_input), events_target = data

        with tf.GradientTape() as tape:
            y_pred = self([events_input, features_input], training=True)
            # x = self.embedder()
            # y_pred = self.compute_input(x)
            seq_lens = K.sum(tf.cast(events_input != 0, dtype=tf.float64), axis=-1)[..., None]
            # sample_weight = class_weight * seq_lens / self.max_len
            sample_weight = None
            # if len(tf.shape(events_target)) == len(tf.shape(y_pred))-1:
            #     events_target = tf.repeat(events_target, self.max_len, axis=-1)[..., None]
            # else:
            #     print("Stop")
            train_loss = self.compiled_loss(
                events_target,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
            # train_loss = K.sum(tf.cast(train_loss, tf.float64)*class_weight)

        trainable_weights = self.trainable_weights
        grads = tape.gradient(train_loss, trainable_weights)
        # tf.print("\n")
        # tf.print(grads[-2])
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        self.compiled_metrics.update_state(events_target, y_pred, sample_weight=sample_weight)
        # trainer_losses = self.custom_loss.composites
        # sanity_losses = self.custom_eval.composites
        # losses = {}
        # # if DEBUG_SHOW_ALL_METRICS:
        # losses.update(trainer_losses)
        # losses.update(sanity_losses)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            (events_input, features_input), events_target, class_weight = data
        else:
            sample_weight = None
            (events_input, features_input), events_target = data  # Compute predictions
        y_pred = self((events_input, features_input), training=False)
        # seq_lens = K.sum(tf.cast(events_input!=0, dtype=tf.float64), axis=-1)[..., None]
        # sample_weight = class_weight # / self.max_len
        # MAYBE THE CULPRIT
        self.compiled_loss(events_target, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(events_target, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        # losses = {}
        # sanity_losses = self.custom_eval.composites
        # losses["loss"] = eval_loss
        # # self.c.append(list(self.custom_eval.composites.values())[0].numpy())
        # tf.print({m.name: m.result() for m in self.metrics})
        # return losses
        return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = loss or self.custom_loss
        metrics = metrics or self.custom_eval
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None):
        events, features = inputs
        x = self.embedder([events, features])
        y_pred = self.compute_input(x)
        return y_pred

    def compute_input(self, x):
        x = self.lstm_layer(x)
        if self.logit_layer is not None:
            x = self.logit_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update({"custom_loss": self.custom_loss, "custom_eval": self.custom_eval})
        return config

    @staticmethod
    def init_metrics():
        return metric.JoinedLoss([metric.MSpCatCE()]), metric.JoinedLoss([metric.MSpCatAcc(), metric.MEditSimilarity()])


class SimpleLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(SimpleLSTM, self).__init__(name=type(self).__name__, **kwargs)
        self.embedder = commons.TokenEmbedderLayer(self.vocab_len, self.embed_dim, mask_zero=0)

    def call(self, inputs, training=None):
        events, features = inputs
        ev_onehot = self.embedder([events, features])
        # x = self.combiner([ev_onehot, features])
        y_pred = self.compute_input(ev_onehot)
        return y_pred


class EmbeddingLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(EmbeddingLSTM, self).__init__(name=type(self).__name__, **kwargs)
        self.embedder = commons.HybridEmbedderLayer(self.vocab_len, self.embed_dim, mask_zero=0)
        # del self.combiner

    def call(self, inputs, training=None):
        x = self.embedder(inputs)
        y_pred = self.compute_input(x)
        return y_pred


class OutcomeLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)
        self.lstm_layer = layers.LSTM(self.ff_dim)
        self.logit_layer = keras.Sequential([layers.Dense(5, activation='tanh'), layers.Dense(1)])
        # self.logit_layer = layers.Dense(1)

        self.activation_layer = layers.Activation('sigmoid')
        self.custom_loss, self.custom_eval = self.init_metrics()

    @staticmethod
    def init_metrics():
        # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
        return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

    def call(self, inputs, training=None):
        return super().call(inputs, training)

# class OutcomeExtensiveLSTM(BaseLSTM):
#     def __init__(self, **kwargs):
#         super(OutcomeExtensiveLSTM, self).__init__(name=type(self).__name__, **kwargs)
#         self.lstm_layer = layers.LSTM(self.ff_dim, return_sequences=True)
#         self.logit_layer = layers.TimeDistributed(layers.Dense(1))
#         self.activation_layer = layers.Activation('sigmoid')
#         self.custom_loss, self.custom_eval = self.init_metrics()

#     @staticmethod
#     def init_metrics():
#         return metric.JoinedLoss([metric.MSpCatCE()]), metric.JoinedLoss([metric.MSpCatAcc(), metric.MEditSimilarity()])
