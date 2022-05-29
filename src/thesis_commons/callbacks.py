import pathlib
from typing import List
import tensorflow as tf
from tensorflow.python.keras import callbacks
from thesis_commons.functions import save_loss, save_metrics
from thesis_commons.functions import create_path
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils

from thesis_commons.constants import PATH_ROOT

class SaveModelImage(callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def on_train_begin(self, logs=None):
        
        tf.keras.utils.plot_model(self.model, to_file=self.filepath, show_shapes=True, show_layer_names=True)
        # utils.plot_model(self.model.build_graph(), to_file=self.filepath, show_shapes=True, show_layer_names=True)

class SerializeLoss(callbacks.Callback):
    def on_train_begin(self, filepath, logs=None):
        save_loss(filepath, self.model.loss)
        save_metrics(filepath, self.model.metrics)
        


class CallbackCollection:

    def __init__(
        self,
        model_name: str,
        models_dir: pathlib.Path,
        is_prod: bool = False,
    ) -> None:
        self.model_name = model_name
        self.models_dir = models_dir
        tmp_chkpt_path = create_path("chkpt_path", self.models_dir / self.model_name)
        self.chkpt_path = tmp_chkpt_path 
        self.tboard_path = create_path("tboard_path", PATH_ROOT / 'logs' / self.model_name)
        self.csv_logger_path = tmp_chkpt_path / "history.csv"
        self.img_path = tmp_chkpt_path / "model.png"
        self.cb_list = []
        self.is_prod = is_prod

    def add(self, cb: callbacks.Callback):
        self.cb_list.append(cb)
        return self

    def build(self):
        # TODO:  Checkpoints only consider val_loss. Make sure it is computed properly.
        self.cb_list.append(callbacks.ModelCheckpoint(filepath=self.chkpt_path, verbose=0 if self.is_prod else 1, save_best_only=self.is_prod))
        self.cb_list.append(callbacks.TensorBoard(log_dir=self.tboard_path))
        self.cb_list.append(callbacks.CSVLogger(filename=self.csv_logger_path))
        self.cb_list.append(SaveModelImage(filepath=self.img_path))
        # self.cb_list.append(SerializeLoss(filepath=self.chkpt_path))
        return self.cb_list





# class SaveCheckpoint(callbacks.ModelCheckpoint):
#     def _save_model(self, epoch, logs):
#         if isinstance(self.model, ModelSaverMixin):
         
#             model: ModelSaverMixin = self.model
#             filepath = pathlib.Path(self.filepath)
#             name = filepath.name
#             logs = logs or {}
#             if model.submodels:
#                 for m in model.submodels:
#                     model.save(name, overwrite=True, options=self._options)
                
#         return super()._save_model(epoch, logs)