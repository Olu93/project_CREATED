import abc
import inspect
from enum import Enum, auto

import tensorflow as tf

from thesis_commons import metric
from thesis_commons.libcuts import (K, layers, losses, metrics, models,
                                    optimizers)
from thesis_commons.modes import TaskModeType


class ModelSaverMixin:
    def __init__(self) -> None:
        super(ModelSaverMixin, self).__init__()
        self.submodels = []


