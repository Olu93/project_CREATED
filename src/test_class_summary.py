# %%

import tensorflow as tf
from tensorflow.python.keras import layers, Model
from thesis_commons import random

# %%

class ALayer(layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.tt = layers.Dense(3)

    def call(self, inputs):
        x = self.tt(inputs)
        return x

class CustomModel(Model):
    def __init__(self) -> None:
        super(CustomModel, self).__init__()
        self.t1 = layers.Dense(5)
        self.t2 = ALayer()
        
    def call(self, inputs):
        x = self.t1(inputs)
        x = self.t2(x)
        return x    

    def build_graph(self):
        events = layers.Input(shape=(177, ), name="events")
        summarizer = Model(inputs=[events], outputs=self.call(events))
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
