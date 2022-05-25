# %%
from pathlib import Path
import tensorflow.python.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2
import abc
from thesis_commons.callbacks import CallbackCollection
from thesis_predictors.models.lstms.lstm import BaseLSTM
from thesis_commons.embedders import EmbedderConstructor
from thesis_commons import metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, TaskModes, FeatureModes
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import OutcomeMockReader as Reader
from thesis_commons.libcuts import optimizers
import os

task_mode = TaskModes.OUTCOME_PREDEFINED
ft_mode = FeatureModes.FULL
epochs = 1
batch_size = 64
adam_init = 0.01
reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
REDUCTION = ReductionV2
train_dataset = reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
val_dataset = reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)
# fa_events[:, -2] = 8
all_models = os.listdir(PATH_MODELS_PREDICTORS)

# %%

# class CustomModel(BaseLSTM):
#     def __init__(self, embed_dim=10, ff_dim=5, **kwargs):
#         super(CustomModel, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
#         ft_mode = kwargs.pop('ft_mode')
#         self.embed_dim = embed_dim
#         self.ff_dim = ff_dim
#         self.embedder = EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, mask_zero=0)
#         self.lstm_layer = keras.layers.LSTM(self.ff_dim)
#         self.logit_layer = keras.Sequential([keras.layers.Dense(5, activation='tanh'), keras.layers.Dense(1)])
#         self.activation_layer = keras.layers.Activation('sigmoid')
#         self.custom_loss, self.custom_eval = self.init_metrics()
#         # self.c = []

#     def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
#         loss = metric.MSpOutcomeCE(), 
#         metrics = [metric.MSpOutcomeAcc()]
#         return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

#     def call(self, inputs, training=None):
#         events, features = inputs
#         x = self.embedder([events, features])
#         y_pred = self.compute_input(x)
#         return y_pred

#     def train_step(self, data):
#         if len(data) == 3:
#             x, events_target, sample_weight = data
#         else:
#             sample_weight = None
#             x, events_target = data


#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             loss = self.compiled_loss(
#                 events_target,
#                 y_pred,
#                 sample_weight=sample_weight,
#                 regularization_losses=self.losses,
#             )

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Update the metrics.
#         # Metrics are configured in `compile()`.
#         self.compiled_metrics.update_state(events_target, y_pred, sample_weight=sample_weight)

#         # Return a dict mapping metric names to current value.
#         # Note that it will include the loss (tracked in self.metrics).
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         # Unpack the data
#         if len(data) == 3:
#             (events_input, features_input), events_target, class_weight = data
#         else:
#             sample_weight = None
#             (events_input, features_input), events_target = data  # Compute predictions
#         y_pred = self((events_input, features_input), training=False)

#         self.compiled_loss(events_target, y_pred, regularization_losses=self.losses)
#         self.compiled_metrics.update_state(events_target, y_pred)
#         return {m.name: m.result() for m in self.metrics}

#     def get_config(self):
#         config = super().get_config()
#         return config

#     @staticmethod
#     def init_metrics():
#         # TODO: RAISES A WARNING -> 5 out of the last 176 calls to <function Model.make_predict_function.<locals>.predict_function at 0x000001FA901F9A60> triggered tf.function retracing.
#         return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()
    
# class OutcomeLSTM(BaseLSTM):
#     def __init__(self, **kwargs):
#         super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)
#         # self.lstm_layer = tf.keras.layers.LSTM(self.ff_dim)
#         # self.logit_layer = keras.Sequential([tf.keras.layers.Dense(5, activation='tanh'), tf.keras.layers.Dense(1)])
#         self.lstm_layer = keras.layers.LSTM(6, return_sequences=True)
#         self.logit_layer = keras.Sequential([keras.layers.Dense(5, activation='tanh'), keras.layers.Dense(1)])
#         self.activation_layer = tf.keras.layers.Activation('sigmoid')
#         self.custom_loss, self.custom_eval = self.init_metrics()

#     @staticmethod
#     def init_metrics():
#         # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
#         return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

#     def call(self, inputs, training=None):
#         return super().call(inputs, training)

# class OutcomeLSTM(BaseLSTM):
#     def __init__(self, **kwargs):
#         super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)
#         self.lstm_layer = tf.keras.layers.LSTM(self.ff_dim)
#         self.logit_layer = keras.Sequential([tf.keras.layers.Dense(5, activation='tanh'), tf.keras.layers.Dense(1)])
#         # self.logit_layer = layers.Dense(1)

#         self.activation_layer = tf.keras.layers.Activation('sigmoid')
#         self.custom_loss, self.custom_eval = self.init_metrics()

#     @staticmethod
#     def init_metrics():
#         # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
#         return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

#     def call(self, inputs, training=None):
#         return super().call(inputs, training)    
    
## %%
test_path = Path("../junk/test_model").absolute()
print(f'Save at {test_path}')
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(verbose=2, filepath=test_path, save_best_only=True)

## %%
# # Construct and compile an instance of CustomModel
model = OutcomeLSTM(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
# model.compile(optimizer="adam", loss=None, metrics=None, run_eagerly=True)

# # You can now use sample_weight argument
# model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=model_checkpoint_callback)

model.build_graph()
model.summary()
model.compile(loss=None, optimizer=optimizers.Adam(adam_init), metrics=None, run_eagerly=True)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=CallbackCollection(model.name, PATH_MODELS_PREDICTORS, True).build())


# %%
# %%
new_model = keras.models.load_model(test_path, custom_objects={obj.name:obj for obj in OutcomeLSTM.init_metrics()})
# %%
(cf_events, cf_features) = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)[0]
(fa_events, fa_features) = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)[0]
new_model.predict((fa_events[1:3], fa_features[1:3])).shape
# new_model.predict((cf_events, cf_features)).shape
# %%
