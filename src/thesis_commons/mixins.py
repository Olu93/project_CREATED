from enum import Enum, auto
import tensorflow as tf
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics
from thesis_commons import metric
from thesis_commons.modes import TaskModeType, InputModeType
import inspect
import abc



class ModelSaverMixin:
    def __init__(self) -> None:
        super(ModelSaverMixin, self).__init__()
        self.submodels = []


