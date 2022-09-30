
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy

from thesis_commons.metric import MEditSimilarity, MSpCatAcc, MSpCatCE
from thesis_commons.modes import TaskModeType

# TODO: Put into thesis_commons package


class ModelInterface(Model):
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    input_interface = TokenInput()
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, **kwargs):
        super(ModelInterface, self).__init__(**kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.kwargs = kwargs
        self.set_metrics()

    def set_metrics(self):
        task_mode_type = self.task_mode_type
        assert task_mode_type is not None, f"Task mode not set. Cannot compile loss or metric. {task_mode_type if not None else 'None'} was given"
        loss_fn = None
        metric_fn = None
        if task_mode_type is TaskModeType.FIX2FIX:
            loss_fn = MSpCatCE()
            metric_fn = [MSpCatAcc(), MEditSimilarity()]
        if task_mode_type is TaskModeType.FIX2ONE:
            # loss_fn = 'sparse_categorical_crossentropy'
            # loss_fn = CustomSpCatCE()
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        if task_mode_type is TaskModeType.MANY2MANY:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        if task_mode_type is TaskModeType.MANY2ONE:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        return self

    def get_config(self):
        config = {
            "vocab_len": self.vocab_len,
            "max_len": self.max_len,
            "feature_len": self.feature_len,
        }
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        metrics = metrics or self.metric_fn
        loss = loss or self.loss_fn
        return super().compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)


    def summary(self):
        return self.input_interface.summary(self)
