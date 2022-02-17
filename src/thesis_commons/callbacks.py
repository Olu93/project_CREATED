import pathlib
from typing import List
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import losses
from tensorflow.keras import callbacks


class CallbackCollection:

    def __init__(
        self,
        model_path: pathlib.Path,
        is_prod: bool = False,
    ) -> None:
        self.model_path = model_path
        self.tboard_path = pathlib.Path('/')
        self.cb_list = []
        self.is_prod = is_prod

    def add(self, cb: callbacks.Callback):
        self.cb_list.append(cb)
        return self

    def build(self):
        self.cb_list.append(callbacks.CSVLogger(filename=self.model_path / "history.csv"))
        self.cb_list.append(callbacks.TensorBoard(log_dir='logs'))
        self.cb_list.append(callbacks.ModelCheckpoint(filepath=self.model_path, verbose=0 if self.is_prod else 1, save_best_only=self.is_prod))
        return self.cb_list
