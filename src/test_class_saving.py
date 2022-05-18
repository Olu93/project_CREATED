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
        self.compute = keras.layers.Dense(1)
        self.compute2 = keras.layers.Dense(1)
        self.lstm_layer = keras.layers.LSTM(6, return_sequences=True)
        self.logit_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_len))
        # TODO: RAISES WARNING -> out of the last 5 calls to
        # self.activation_layer = keras.layers.Activation('sigmoid')

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = metric.MSpOutcomeCE(), 
        metrics = [metric.MSpOutcomeAcc()]
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        x, z = inputs
        return self.compute(x)[...,None] + self.compute2(z)


    def train_step(self, data):
        # print("Train-Step")
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) >= 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            # print(data[0][0].shape)
            # print(data[0][1].shape)
            # print(len(data))
            # print(len(data[0]))
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            # print("y.shape")
            # print(y.shape)
            # print("y_pred.shape")
            # print(y_pred.shape)
            loss = self.compiled_loss(
                y,
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
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # print("Test-Step")
        # Unpack the data
        if len(data) >= 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            # print(data[0][0].shape)
            # print(data[0][1].shape)
            # print(len(data))
            # print(len(data[0]))
            x, y = data        
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        return config

    @staticmethod
    def init_metrics():
        # TODO: RAISES A WARNING -> 5 out of the last 176 calls to <function Model.make_predict_function.<locals>.predict_function at 0x000001FA901F9A60> triggered tf.function retracing.
        return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()
    

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
