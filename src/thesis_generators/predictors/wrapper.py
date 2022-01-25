from typing import List
import numpy as np
from tensorflow.keras import Model
from thesis_readers import VolvoIncidentsReader, RequestForPaymentLogReader, BPIC12LogReader, AbstractProcessLogReader
from tensorflow import keras
import tensorflow as tf
import pathlib
import os
import io

from thesis_predictors.helper import metrics
from thesis_predictors.helper.constants import MODEL_FOLDER


class ModelWrapper():
    prediction_model: Model = None

    def __init__(self, reader: AbstractProcessLogReader, model_num: int = None, model_name: str = None) -> None:
        self.reader = reader
        self.model_dirs = {x.name: x for x in MODEL_FOLDER.iterdir() if x.is_dir()}
        if model_num:
            self.load_model_by_num(model_num)
        if model_name:
            self.load_model_by_name(model_name)

    def load_model_by_path(self, model_path: pathlib.Path):
        self.model_path = model_path
        self.prediction_model = keras.models.load_model(self.model_path,
                                                        custom_objects={
                                                            'MaskedSpCatCE': metrics.MaskedSpCatCE(),
                                                            'MaskedSpCatAcc': metrics.MaskedSpCatAcc(),
                                                            'MaskedEditSimilarity': metrics.MaskedEditSimilarity(),
                                                        })
        self.model_name = self.prediction_model.name
        return self

    def load_model_by_num(self, model_num: int):
        self.model_num = model_num
        chosen_model = list(self.model_dirs.items())[self.model_num][1]
        return self.load_model_by_path(chosen_model)

    def load_model_by_name(self, model_name: str):
        chosen_model = self.model_dirs[model_name]
        return self.load_model_by_path(chosen_model)

    def prepare_input(self, example):
        structure = self.reader.get_dataset()._structure[0]

        shape_batch = (example.shape[0], )
        return (
            tf.constant(example, dtype=tf.float32),
            tf.zeros(shape_batch + structure[1].shape[1:]),
            tf.zeros(shape_batch + structure[2].shape[1:]),
            tf.zeros(shape_batch + structure[3].shape[1:]),
        )

    def predict_sequence(self, sequence) -> np.ndarray:
        sequence = sequence[None] if sequence.ndim < 2 else sequence
        input_for_prediction = self.prepare_input(sequence)
        return self.prediction_model.predict(input_for_prediction)
