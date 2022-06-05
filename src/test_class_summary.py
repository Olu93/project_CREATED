# %%
import abc
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader as Reader
from thesis_commons.libcuts import random
# %%

class ALayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.tt = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.tt(inputs)
        return x

class CustomModel(tf.keras.Model):
    def __init__(self) -> None:
        super(CustomModel, self).__init__()
        self.t1 = tf.keras.layers.Dense(5)
        self.t2 = ALayer()
        
    def call(self, inputs):
        x = self.t1(inputs)
        x = self.t2(x)
        return x    

    def build_graph(self):
        events = tf.keras.layers.Input(shape=(177, ), name="events")
        summarizer = tf.keras.Model(inputs=[events], outputs=self.call(events))
        return summarizer

x_data = random.random((32, 177))
z_data = random.random((32, 177))
model = CustomModel()
model.compile()
model = model.build_graph()
# model.build_graph()
# model.summary()
res = model.predict(x_data)
res
# model
# new_model.predict((cf_events, cf_features)).shape
# %%
