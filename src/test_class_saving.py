# %%
from pathlib import Path
import tensorflow.python.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2
import abc
from thesis_predictors.models.lstms.lstm import BaseLSTM
from thesis_generators.models.model_commons import TensorflowModelMixin
from thesis_generators.models.model_commons import HybridInput
from thesis_generators.models.model_commons import EmbedderConstructor
from thesis_commons import metric
from thesis_generators.models.model_commons import InputInterface
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, TaskModes, FeatureModes
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader as Reader
import os

task_mode = TaskModes.OUTCOME_PREDEFINED
ft_mode = FeatureModes.FULL_SEP
epochs = 2
batch_size = 64
reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
REDUCTION = ReductionV2
train_dataset = reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
val_dataset = reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)
# fa_events[:, -2] = 8
all_models = os.listdir(PATH_MODELS_PREDICTORS)

# %%

class CustomModel(BaseLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=10, mask_zero=0)
        self.lstm_layer = keras.layers.LSTM(6)
        self.logit_layer = keras.Sequential([keras.layers.Dense(5, activation='tanh'), keras.layers.Dense(1)])
        self.activation_layer = keras.layers.Activation('sigmoid')
        self.custom_loss, self.custom_eval = self.init_metrics()

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
        # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
        return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

    def train_step(self, data):
        if len(data) == 3:
            (events_input, features_input), events_target, sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), events_target = data

        with tf.GradientTape() as tape:
            y_pred = self([events_input, features_input], training=True)
            # x = self.embedder()
            # y_pred = self.compute_input(x)
            seq_lens = keras.backend.sum(tf.cast(events_input != 0, dtype=tf.float64), axis=-1)[..., None]
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
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        self.compiled_metrics.update_state(events_target, y_pred, sample_weight=sample_weight)
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
        return {m.name: m.result() for m in self.metrics}



class OutcomeLSTM(BaseLSTM):
    def __init__(self, **kwargs):
        super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)
        # self.lstm_layer = tf.keras.layers.LSTM(self.ff_dim)
        # self.logit_layer = keras.Sequential([tf.keras.layers.Dense(5, activation='tanh'), tf.keras.layers.Dense(1)])
        self.lstm_layer = keras.layers.LSTM(6, return_sequences=True)
        self.logit_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_len))
        self.activation_layer = tf.keras.layers.Activation('sigmoid')
        self.custom_loss, self.custom_eval = self.init_metrics()

    @staticmethod
    def init_metrics():
        # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
        return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

    def call(self, inputs, training=None):
        return super().call(inputs, training)
    
## %%
test_path = Path("./junk/test_model").absolute()
print(f'Save at {test_path}')
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(verbose=2, filepath=test_path, save_best_only=True)

## %%
# Construct and compile an instance of CustomModel
model = CustomModel(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
model.compile(optimizer="adam", loss=None, metrics=None, run_eagerly=True)

# You can now use sample_weight argument
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=model_checkpoint_callback)

# %%
# %%
new_model = keras.models.load_model(test_path, custom_objects={obj.name:obj for obj in CustomModel.init_metrics()})
# %%
(cf_events, cf_features) = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)[0]
(fa_events, fa_features) = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)[0]
new_model.predict((fa_events[1:3], fa_features[1:3])).shape
# new_model.predict((cf_events, cf_features)).shape
# %%
