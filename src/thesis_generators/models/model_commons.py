import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from thesis_predictors.helper.metrics import MaskedEditSimilarity, MaskedSpCatCE, MaskedSpCatAcc
from thesis_predictors.models.model_commons import TokenInput
from thesis_readers.helper.modes import TaskModeType, InputModeType

class GeneratorInterface():
    # def __init__(self) -> None:
    task_mode_type: TaskModeType = None
    input_interface = TokenInput()
    loss_fn: Loss = None
    metric_fn: Metric = None

    def __init__(self, vocab_len, max_len, feature_len, **kwargs):
        super(GeneratorInterface, self).__init__(**kwargs)
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
            loss_fn = MaskedSpCatCE()
            metric_fn = [MaskedSpCatAcc(), MaskedEditSimilarity()]
        return self