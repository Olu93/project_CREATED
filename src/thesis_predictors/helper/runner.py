import io
from typing import Type
from thesis_commons.callbacks import CallbackCollection
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from keras.api._v2.keras.models import Model
import tqdm
import json
from thesis_commons.libcuts import optimizers
import pathlib
from thesis_commons.modes import FeatureModes, DatasetModes
from thesis_readers import AbstractProcessLogReader
import tensorflow as tf
import thesis_commons.model_commons as commons

# TODO: Put in runners module. This module is a key module not a helper.
DEBUG = True

class Runner(object):
    statistics = {}

    def __init__(
            self,
            Model: Type[commons.TensorflowModelMixin],
            reader: AbstractProcessLogReader,
            epochs: int,
            batch_size: int,
            adam_init: float,
            num_train: int = None,
            num_val: int = None,
            num_test: int = None,
            ft_mode: FeatureModes = FeatureModes.FULL,
            **kwargs,
    ):
        self.reader = reader
        self.train_dataset = self.reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
        self.val_dataset = self.reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)
        self.test_dataset = self.reader.get_dataset_with_indices(DatasetModes.TEST, ft_mode=ft_mode)
        self.model = Model(vocab_len=self.reader.vocab_len, max_len=self.reader.max_len, feature_len=self.reader.current_feature_len, ft_mode=ft_mode, **kwargs)

        if num_train:
            self.train_dataset = self.train_dataset.take(num_train)
        if num_val:
            self.val_dataset = self.val_dataset.take(num_val)
        if num_test:
            self.test_dataset = self.test_dataset.take(num_test) # TODO: IMPORTANT FIX - Was the wrong parameter!!!!
        

        self.epochs = epochs
        self.batch_size = batch_size
        self.adam_init = adam_init
        self.start_id = reader.start_id
        self.end_id = reader.end_id

        self.label = self.model.name

    def train_model(self, label=None, train_dataset=None, val_dataset=None):
        label = label or self.label
        train_dataset = train_dataset or self.train_dataset
        val_dataset = val_dataset or self.val_dataset
        # self.metrics = metrics
        # self.loss_fn = loss_fn

        print(f"{label}:")
        # TODO: Impl: check that checks whether ft_mode is compatible with model feature type
        self.model.build_graph()
        self.model.summary()
        self.model.compile(loss=None, optimizer=optimizers.Adam(self.adam_init), metrics=None, run_eagerly=DEBUG)

        self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs, callbacks=CallbackCollection(self.model.name, PATH_MODELS_PREDICTORS, DEBUG).build())

        return self

    def evaluate(self, evaluator, save_path="results", prefix="full", label=None, test_dataset=None, dont_save=False):
        test_dataset = test_dataset or self.test_dataset
        test_dataset = self.reader.gather_full_dataset(self.test_dataset)
        self.results = evaluator.set_model(self.model).evaluate(test_dataset)
        if not dont_save:
            label = label or self.label
            save_path = save_path or self.save_path
            self.results.to_csv(pathlib.Path(save_path) / (f"{prefix}_{label}.csv"))
        return self

    # def save_model(self, save_path="build", prefix="full", label=None):
    #     label = label or self.label
    #     save_path = save_path or self.save_path
    #     target_folder = pathlib.Path(save_path) / (f"{prefix}_{label}")
    #     self.model.save(target_folder)
    #     self.model_path = target_folder
    #     json.dump(self._transform_model_history(), io.open(target_folder / 'history.json', 'w'), indent=4, sort_keys=True)
    #     return self

    # def _transform_model_history(self):
    #     tmp_history = dict(self.history.history)
    #     tmp_history["epochs"] = self.history.epoch
    #     history = {
    #         "history": tmp_history,
    #         "params": self.history.params,
    #     }
    #     return history