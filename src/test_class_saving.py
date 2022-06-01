# %%
import abc
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.utils.losses_utils import ReductionV2

import thesis_commons.model_commons as commons
# from thesis_predictors.models.lstms.lstm import BaseLSTM
from thesis_commons import embedders as embedders
from thesis_commons import metric
from thesis_commons.callbacks import CallbackCollection
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.libcuts import optimizers
from thesis_commons.modes import (DatasetModes, FeatureModes, TaskModes,
                                  TaskModeType)
# from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import OutcomeMockReader as Reader

task_mode = TaskModes.OUTCOME_PREDEFINED
ft_mode = FeatureModes.FULL
epochs = 1
batch_size = 64
adam_init = 0.01
reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
REDUCTION = ReductionV2
# train_dataset = reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
# val_dataset = reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)
train_dataset = reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
val_dataset = reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)

(tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
(fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)

# %%


# class BaseLSTM(commons.TensorflowModelMixin):
#     task_mode_type = TaskModeType.FIX2FIX

#     def __init__(self, ft_mode, embed_dim=10, ff_dim=5, **kwargs):
#         super(BaseLSTM, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
#         self.embed_dim = embed_dim
#         self.ff_dim = ff_dim
#         ft_mode = ft_mode
#         self.embedder = embedders.EmbedderConstructor(ft_mode=ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, mask_zero=0)
#         self.lstm_layer = tf.keras.layers.LSTM(self.ff_dim, return_sequences=True)
#         self.logit_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_len))
#         self.activation_layer = tf.keras.layers.Activation('softmax')
#         self.custom_loss, self.custom_eval = self.init_metrics()
#         # self.c = []

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

#     def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
#         loss = loss or self.custom_loss
#         metrics = metrics or self.custom_eval
#         return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

#     def call(self, inputs, training=None):
#         events, features = inputs
#         x = self.embedder([events, features])
#         y_pred = self.compute_input(x)
#         return y_pred

#     def compute_input(self, x):
#         x = self.lstm_layer(x)
#         if self.logit_layer is not None:
#             x = self.logit_layer(x)
#         y_pred = self.activation_layer(x)
#         return y_pred

#     def get_config(self):
#         config = super().get_config()
#         config.update({"custom_loss": self.custom_loss, "custom_eval": self.custom_eval})
#         return config

#     def build_graph(self) -> tf.keras.Model:
#         events = tf.keras.layers.Input(shape=(self.max_len, ), name="events")
#         features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len), name="event_attributes")
#         inputs = [events, features]
#         summarizer = tf.keras.models.Model(inputs=[inputs], outputs=self.call(inputs))
#         # return summarizer
#         # self(inputs)
#         self.call(inputs)
#         return summarizer

#     @staticmethod
#     def init_metrics():
#         return metric.JoinedLoss([metric.MSpCatCE()]), metric.JoinedLoss([metric.MSpCatAcc(), metric.MEditSimilarity()])


# class OutcomeLSTM(BaseLSTM):
#     def __init__(self, **kwargs):
#         super(OutcomeLSTM, self).__init__(name=type(self).__name__, **kwargs)
#         self.lstm_layer = tf.keras.layers.LSTM(self.ff_dim)
#         self.logit_layer = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='tanh'), tf.keras.layers.Dense(1)])
#         # self.logit_layer = layers.Dense(1)
#         # self.embedder = tf.keras.layers.Embedding(self.vocab_len, output_dim=30)

#         self.activation_layer = tf.keras.layers.Activation('sigmoid')
#         self.custom_loss, self.custom_eval = self.init_metrics()

#     @staticmethod
#     def init_metrics():
#         # return metric.JoinedLoss([metric.MSpOutcomeCE()]), metric.JoinedLoss([metric.MSpOutcomeAcc()])
#         return metric.MSpOutcomeCE(), metric.MSpOutcomeAcc()

    # def call(self, inputs, training=None):
    #     x, y = inputs 
    #     x = self.embedder(x)
    #     x = self.lstm_layer(x)
    #     if self.logit_layer is not None:
    #         x = self.logit_layer(x)
    #     y_pred = self.activation_layer(x)        
    #     return y_pred


## %%
# test_path = Path("../junk/test_model").absolute()
# print(f'Save at {test_path}')
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(verbose=2, filepath=test_path, save_best_only=True)

## %%
# # Construct and compile an instance of CustomModel
model = BaseLSTM(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
# model.compile(optimizer="adam", loss=None, metrics=None, run_eagerly=True)

# # You can now use sample_weight argument
# model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=model_checkpoint_callback)

# model.build_graph()
# model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizers.Adam(adam_init), metrics=tf.keras.metrics.BinaryAccuracy(), run_eagerly=True)

PATH_MODELS_PREDICTORS_CHKPT = Path("../junk/test_chkpt_model").absolute()
PATH_MODELS_PREDICTORS_FINAL = Path("../junk/test_final_model").absolute()
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=CallbackCollection(model.name, PATH_MODELS_PREDICTORS_CHKPT, True).build())
model.save(PATH_MODELS_PREDICTORS_FINAL / model.name, True)

# %%
# custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
chkpt_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS_CHKPT / model.name)
final_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS_FINAL / model.name)
# custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
# chkpt_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS_CHKPT / model.name, custom_objects=custom_objects_predictor)
# final_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS_FINAL / model.name, custom_objects=custom_objects_predictor)
# %%
def test_model(fa_events, fa_features, model, message):
    print("=========================")
    try:
        print(model.predict([fa_events, fa_features]).shape)
    except Exception as e:
        print(message)
        print(e)


test_model(fa_events, fa_features, model, "Unsaved did not work!")
test_model(fa_events, fa_features, chkpt_model, "Chkpt did not work!")
test_model(fa_events, fa_features, final_model, "Final did not work!")
# %%
# TODO: https://keras.io/guides/serialization_and_saving/#:~:text=Registering%20the%20custom%20object&text=Keras%20keeps%20a%20master%20list,Value%20Error%3A%20Unknown%20layer%20).
# TODO: https://keras.io/api/utils/serialization_utils/#registerkerasserializable-function
# TODO: https://keras.io/api/utils/serialization_utils/#customobjectscope-class