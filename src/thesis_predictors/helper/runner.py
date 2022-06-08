

import thesis_commons.model_commons as commons
from thesis_commons.callbacks import CallbackCollection
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.libcuts import optimizers
from thesis_readers import AbstractProcessLogReader

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

        # self.start_id = reader.start_id DELETE
        # self.end_id = reader.end_id

        self.label = self.model.name

    def train_model(
        self,
        train_dataset=None,
        val_dataset=None,
        epochs: int=50,
        adam_init: float=None,
        label=None,
    ):

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

        self.history = self.model.fit(train_dataset,
                                      validation_data=val_dataset,
                                      epochs=epochs,
                                      callbacks=CallbackCollection(self.model.name, PATH_MODELS_PREDICTORS, DEBUG).build())

        return self

    # def evaluate(self, evaluator, save_path="results", prefix="full", label=None, test_dataset=None, dont_save=False):
    #     test_dataset = test_dataset or self.test_dataset
    #     test_dataset = self.reader.gather_full_dataset(self.test_dataset)
    #     self.results = evaluator.set_model(self.model).evaluate(test_dataset)
    #     if not dont_save:
    #         label = label or self.label
    #         save_path = save_path or self.save_path
    #         self.results.to_csv(pathlib.Path(save_path) / (f"{prefix}_{label}.csv"))
    #     return self

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