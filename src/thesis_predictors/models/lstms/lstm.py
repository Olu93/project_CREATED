from typing import ClassVar, Generic, Type, TypeVar
import tensorflow as tf
from thesis_readers.readers.OutcomeReader import OutcomeMockReader
from thesis_commons.modes import DatasetModes, FeatureModes
from thesis_readers.readers.MockReader import MockReader
from thesis_commons.constants import REDUCTION
from thesis_commons.modes import TaskModeType
from thesis_commons.libcuts import layers, K, losses, keras, optimizers
import thesis_commons.model_commons as commons
import thesis_commons.embedders as embedders 
# TODO: import thesis_commons.model_commons as commons
from thesis_commons import metric

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
DEBUG_SHOW_ALL_METRICS = False
# TODO: Think of double stream LSTM: One for features and one for events.
# Both streams are dependendant on previous features and events.
# Requires very special loss that takes feature differences and event categorical loss into account


class BaseLSTM(commons.TensorflowModelMixin):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, embed_dim=10, ff_dim=5, **kwargs):
        super(BaseLSTM, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        ft_mode = kwargs.pop('ft_mode')
        self.embedder = embedders.EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, mask_zero=0)
        self.lstm_layer = layers.LSTM(self.ff_dim, return_sequences=True)
        self.logit_layer = layers.TimeDistributed(layers.Dense(self.vocab_len))
        self.activation_layer = layers.Activation('softmax')
        self.custom_loss, self.custom_eval = self.init_metrics()
        # self.c = []

    def train_step(self, data):
        if len(data) == 3:
            x, events_target, sample_weight = data
        else:
            sample_weight = None
            x, events_target = data


        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                events_target,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(events_target, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            (events_input, features_input), events_target, class_weight = data
        else:
            sample_weight = None
            (events_input, features_input), events_target = data  # Compute predictions
        y_pred = self((events_input, features_input), training=False)

        self.compiled_loss(events_target, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(events_target, y_pred)
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

if __name__ == "__main__":
    reader = OutcomeMockReader().init_log().init_meta(False)
    epochs = 1
    adam_init = 0.001
    ft_mode = FeatureModes.EVENT
    print("Simple LSTM Mono:")
    data = reader.get_dataset(data_mode=DatasetModes.TRAIN, ft_mode=ft_mode)
    model = SimpleLSTM(ft_mode=ft_mode, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len)
    model.compile(loss=model.loss_fn, optimizer=optimizers.Adam(adam_init), metrics=model.metrics)
    model = model.build_graph()
    model.summary()