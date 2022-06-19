from __future__ import annotations
from abc import ABC, abstractmethod
from collections import UserList
from typing import Callable, Dict, List
import pandas as pd


class Operation(ABC):
    name = None
    i_cols = None
    o_cols = None
    _children: List[Operation] = None
    _parents: List[Operation] = None
    _params: Dict = None
    _params_r: Dict = None

    def __init__(self, forward: Callable, backward: Callable, **kwargs):
        self.forward = forward
        self.backward = backward

    @abstractmethod
    def process(self, inputs, **kwargs):
        self._params = kwargs
        result, params_r = self.forward(inputs, **kwargs)
        self._params_r = params_r
        return result

    @abstractmethod
    def revert(self, inputs, **kwargs):
        self._params = kwargs
        result, params_r = self.backward(inputs, **kwargs)
        self._params_r = params_r
        return result


class ReversableOperation(Operation):
    pass


class IrreversableOperation(Operation):
    pass

class DropOperation(IrreversableOperation):
    pass


class ProcessingPipeline(UserList):
    pass