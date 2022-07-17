import pathlib

import thesis_commons.model_commons as commons
from thesis_commons.callbacks import CallbackCollection
from thesis_commons.constants import PATH_MODELS_GENERATORS
import tensorflow as tf

keras = tf.keras
from keras import optimizers
from thesis_readers import AbstractProcessLogReader
import numpy as np
# TODO: Put in runners module. This module is a key module not a helper.
DEBUG = True


class Runner(object):
    statistics = {}

    def __init__(
        self,
        model: commons.TensorflowModelMixin,
        reader: AbstractProcessLogReader,
        **kwargs,
    ):
        self.reader = reader
        self.model = model

        self.start_id = reader.start_id
        self.end_id = reader.end_id

        self.label = self.model.name

    def train_model(self, train_dataset=None, val_dataset=None, epochs: int = 50, adam_init: float = None, label=None, skip_callbacks=False):

        label = label or self.label
        train_dataset = train_dataset
        val_dataset = val_dataset
        # self.metrics = metrics
        # self.loss_fn = loss_fn

        print(f"{label}:")
        # TODO: Impl: check that checks whether ft_mode is compatible with model feature type
        self.model.compile(loss=None, optimizer=optimizers.Adam(adam_init), metrics=None, run_eagerly=DEBUG)
        x_pred, y_true = next(iter(train_dataset))
        y_pred = self.model(x_pred)
        self.model.summary()

        callbacks = CallbackCollection(self.model.name, PATH_MODELS_GENERATORS, DEBUG).build() if not skip_callbacks else None
        self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)

        return self

    def evaluate(self, test_dataset, k_fa=5):
        print(f"============================= START Quick Eval Completed: {self.model.name} =============================")
        test_dataset = self.reader.gather_full_dataset(test_dataset)
        x_ev, x_ft, y_ev, y_ft = test_dataset
        x_ev, x_ft, y_ev, y_ft = x_ev[:k_fa], x_ft[:k_fa], y_ev[:k_fa], y_ft[:k_fa]
        # input_data = tf.data.Dataset.from_tensor_slices((x_ev, x_ft))
        y_pred_ev, y_pred_ft = self.model((x_ev, x_ft))
        print(x_ev)
        print(y_pred_ev)
        print(y_ev)
        print(f"Quick Eval Completed\n")
        return self
