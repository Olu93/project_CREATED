from tensorflow.keras import Model
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy

from thesis_readers.helper.modes import TaskModes
from ..helper.metrics import SparseCrossEntropyLoss, SparseAccuracyMetric
from enum import IntEnum, auto, Enum


class ModelInterface(Model):
    # def __init__(self) -> None:
    loss_fn = None
    metric_fn = None

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_metrics(kwargs.pop('task_mode', None))

    def set_metrics(self, task_mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE):
        loss_fn = None
        metric_fn = None
        if task_mode in [TaskModes.NEXT_EVENT_EXTENSIVE, TaskModes.OUTCOME_EXTENSIVE]:
            loss_fn = SparseCrossEntropyLoss()
            metric_fn = [SparseAccuracyMetric()]
        if task_mode in [TaskModes.NEXT_EVENT, TaskModes.OUTCOME]:
            loss_fn = SparseCategoricalCrossentropy()
            metric_fn = [SparseCategoricalAccuracy()]
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        return self

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
