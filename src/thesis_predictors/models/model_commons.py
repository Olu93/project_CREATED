from tensorflow.keras import Model
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy

from thesis_readers.readers.AbstractProcessLogReader import TaskModes
from ..helper.metrics import SparseCrossEntropyLossExtensive, SparseAccuracyMetricExtensive


class ModelInterface(Model):
    # def __init__(self) -> None:
    #     self.loss_fn = SparseCrossEntropyLoss()
    #     self.metric_fn = SparseAccuracyMetric()

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_metrics(self, task_mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE):
        if task_mode in [TaskModes.NEXT_EVENT_EXTENSIVE, TaskModes.OUTCOME_EXTENSIVE]:
            loss_fn = SparseCrossEntropyLossExtensive()
            metric_fn = [SparseAccuracyMetricExtensive()]
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
