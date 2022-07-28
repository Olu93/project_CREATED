import pathlib

import tensorflow as tf
keras = tf.keras
from keras import callbacks, utils
import visualkeras
from thesis_commons.constants import PATH_ROOT
from thesis_commons.functions import create_path, save_loss, save_metrics
from thesis_commons.model_commons import TensorflowModelMixin

class SaveModelImage(callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def on_train_begin(self, logs=None):
        model:TensorflowModelMixin = self.model
        model_built = model.build_graph()
        utils.plot_model(model_built, to_file=self.filepath, show_shapes=True, show_layer_names=True)

# https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network
class SaveModelImageVisualkeras(callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def on_train_begin(self, logs=None):
        model:TensorflowModelMixin = self.model
        model_built = model.build_graph()
        visualkeras.layered_view(model_built, to_file=self.filepath, legend=True, draw_volume=False)

class SerializeLoss(callbacks.Callback):
    def on_train_begin(self, filepath, logs=None):
        model:TensorflowModelMixin = self.model
        save_loss(filepath, model.loss)
        save_metrics(filepath, model.metrics)
        


class CallbackCollection:

    def __init__(
        self,
        model_name: str,
        models_dir: pathlib.Path,
        is_debug: bool = False,
    ) -> None:
        self.model_name = model_name
        self.models_dir = models_dir
        tmp_chkpt_path = create_path("chkpt_path", self.models_dir / self.model_name)
        self.chkpt_path = tmp_chkpt_path 
        self.tboard_path = create_path("tboard_path", PATH_ROOT / 'logs' / self.model_name)
        self.csv_logger_path = tmp_chkpt_path / "history.csv"
        self.img_path1 = tmp_chkpt_path / "model_keras_utils.png"
        self.img_path2 = tmp_chkpt_path / "model_visual_keras.png"
        self.cb_list = []
        self.is_prod = not is_debug

    def add(self, cb: callbacks.Callback):
        self.cb_list.append(cb)
        return self

    def build(self, skip_checkpoint=False):
        # TODO:  Checkpoints only consider val_loss. Make sure it is computed properly.
        self.cb_list.append(callbacks.TerminateOnNaN())
        if not skip_checkpoint:
            self.cb_list.append(callbacks.ModelCheckpoint(filepath=self.chkpt_path, verbose=0 if self.is_prod else 1, save_best_only=self.is_prod))
        self.cb_list.append(callbacks.TensorBoard(log_dir=self.tboard_path))
        self.cb_list.append(callbacks.CSVLogger(filename=self.csv_logger_path))
        self.cb_list.append(SaveModelImage(filepath=self.img_path1))
        # self.cb_list.append(SaveModelImageVisualkeras(filepath=self.img_path2))
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