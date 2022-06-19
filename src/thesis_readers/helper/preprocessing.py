from __future__ import annotations
from abc import ABC, abstractmethod
from collections import UserList
from typing import Callable, Dict, List
import pandas as pd

from thesis_commons.representations import BetterDict





class Operation(ABC):
    name = None
    i_cols = None
    o_cols = None
    _children: List[Operation] = None
    _parents: List[Operation] = None
    _params: BetterDict = None
    _params_r: BetterDict = None#

    def __init__(self, forward: Callable, backward: Callable, **kwargs:BetterDict):
        self.forward = forward
        self.backward = backward
        self._params = kwargs or BetterDict()

    def set_params(self, **kwargs)-> Operation:
        self._params = kwargs
        return self

    def add_child(self, child: Operation)-> Operation:
        self._children.append(child)
        return self

    def add_parent(self, parent: Operation)-> Operation:
        self._parents.append(parent)
        return self

    @abstractmethod
    def process(self, inputs, **kwargs):
        self._params = self._params.merge(kwargs)
        result, params_r = self.forward(inputs, **self._params)
        self._params_r = params_r
        self.result = result
        return self

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
    def __init__(self, forward: Callable, backward: Callable=None, **kwargs):
        super().__init__(forward, backward, **kwargs)
    
    def drop(self, data: pd.DataFrame, cols=None):
        self._params = {"cols":cols}
        if not cols:
            self.new_data = data
            return None
        new_data = data.drop(cols, axis=1)
        # removed_cols = set(data.columns) - set(new_data.columns)
        self.new_data = new_data
        self._params_r = new_data.columns
        

class ProcessingPipeline(UserList):
    pass