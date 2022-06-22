from __future__ import annotations
from abc import ABC, abstractmethod
from collections import UserList
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn import preprocessing

from thesis_commons.representations import BetterDict


class Mapping:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return f"Mapping[{self.key} -> {self.value}]"


class Operation(ABC):
    def __init__(self, name: str = "default", digest_fn: Callable = None, revert_fn: Callable = None, **kwargs: BetterDict):
        self.i_cols = None
        self.o_cols = None
        self._next: List[Operation] = []
        self._prev: List[Operation] = []
        self._params: BetterDict = BetterDict()
        self._params_r: BetterDict = BetterDict()  #
        self._digest = digest_fn
        self._revert = revert_fn
        self.name = name
        self._params = kwargs or BetterDict()

    def set_params(self, **kwargs) -> Operation:
        self._params = kwargs
        return self

    def append_next(self, child: Operation, **kwargs) -> Operation:
        self._next.append(child.append_prev(self))
        return self

    def chain(self, child: Operation, **kwargs) -> Operation:
        c = child.append_prev(self)
        self._next.append(c)
        return c

    def append_prev(self, parent: Operation, **kwargs) -> Operation:
        self._prev.append(parent)
        self._params = self._params.copy().merge(parent._params).merge(kwargs)
        return self

    # @abstractmethod
    def forward(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, BetterDict]:
        if self.name == 'label_encode':
            print("Stop")
        if self.name == 'time_extraction':
            print("Stop")
        self._params = BetterDict(self._params).merge(kwargs)
        post_data, params_r = self.digest(data, **self._params)

        # if not len(self._children):
        #     self._result = data.copy()
        #     self._params_r = kwargs.copy()
        #     return self._result, self._params_r

        params_r_collector = BetterDict(params_r)
        for child in self._next:
            post_data, params_r = child.forward(post_data, **self._params)
            params_r_collector[child.name] = params_r

        self._result = post_data.copy()
        self._params_r = params_r_collector.copy()
        return self._result, self._params_r

    # @abstractmethod
    def backward(self, data: pd.DataFrame, **kwargs):
        self._params = kwargs
        result, params_r = self.revert(data, **kwargs)
        self._params_r = params_r
        return result

    @property
    def result(self) -> pd.DataFrame:
        return self._result

    def digest(self, data: pd.DataFrame, **kwargs):
        return self._digest(data, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs):
        return self._revert(data, **kwargs)


class ReversableOperation(Operation):
    def digest(self, data: pd.DataFrame, **kwargs):
        return super().digest(data, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs):
        return super().revert(data, **kwargs)

    def _select_cols(self, col: str, stats: Dict, data: pd.DataFrame) -> bool:
        return True


class IrreversableOperation(Operation):
    def digest(self, data: pd.DataFrame, **kwargs):
        return super().digest(data, **kwargs)

    def revert(self, data: pd.DataFrame, **kwargs):
        return super().digest(data, **kwargs)


class ToDatetimeOperation(IrreversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols = cols

    def digest(self, data: pd.DataFrame, **kwargs):
        data[self.cols] = pd.to_datetime(data[self.cols])
        return data, {}

class DropOperation(IrreversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols = cols
        self.pre2post = {}
        self.post2pre = None
        
    def digest(self, data: pd.DataFrame, **kwargs):
        new_data = data.drop(self.cols, axis=1)
        self.pre2post = {col: col in new_data.columns for col in self.cols}
        return new_data, {}

class BinaryEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.pre2post = {}
        self.post2pre = {}
        self.cols = cols
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        if not len(self.cols):
            return data, {}
        encoder = preprocessing.OneHotEncoder(drop='if_binary', sparse=False)
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data[self.cols] = new_data
        self.pre2post = {col: [col] for col in self.cols}
        self.post2pre = {col: col for col in self.cols}
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.post2pre.keys
        data[keys] = self.encoder.inverse_transform(data[keys])
        return data, {}


class CategoryEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.pre2post = {}
        self.post2pre = {}
        self.cols = cols
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        if not len(self.cols):
            return data, {}
        encoder = ce.BaseNEncoder(return_df=True, drop_invariant=True, base=2)
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data = pd.concat([data, new_data], axis=1)
        for col in self.cols:
            result_cols = [ft for ft in encoder.get_feature_names() if col in ft]
            self.pre2post = {**self.pre2post, col: result_cols}
            self.post2pre = {**self.post2pre, **{rcol: col for rcol in result_cols}}
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = np.unique(list(self.pre2post.keys))
        for k in keys:
            mappings = self.pre2post[k]
            values = [m.value for m in mappings]
            data[k] = self.encoder.inverse_transform(data[values]).drop(values, axis=1)
        return data, {}


class NumericalEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.pre2post = None
        self.post2pre = None
        self.cols = cols
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        if not len(self.cols):
            return data, {}
        encoder = preprocessing.StandardScaler()
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data[self.cols] = new_data
        self.pre2post = {col: [col] for col in self.cols}
        self.post2pre = {col: col for col in self.cols}
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.post2pre.keys
        data[keys] = self.encoder.inverse_transform(data[keys])
        return data, {}


class LabelEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols = cols
        self.col2pp: Dict[str, preprocessing.LabelEncoder] = {}

    def digest(self, data: pd.DataFrame, **kwargs):
        for col in self.cols:
            if col not in data:
                continue
            self.col2pp[col] = preprocessing.LabelEncoder().fit(data[col])
            data[col] = self.col2pp[col].transform(data[col])
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        for col, preprocessor in self.col2pp.items():
            data[col] = preprocessor.inverse_transform(data[col])
        return data, {}

    # def _select_cols(self, col: str, stats: Dict, data: pd.DataFrame) -> bool:
    #     is_encodable = stats.get("is_categorical") or stats.get("is_binary")
    #     is_not_skippable = any([stats.get("is_col_case_id"), stats.get("is_col_outcome")])
    #     is_in_data = col in data.columns
    #     return is_encodable and is_not_skippable and is_in_data


class TimeExtractOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols = cols
        self.pre2post = {}
        self.post2pre = {}

    def digest(self, data: pd.DataFrame, **kwargs):
        time_data = pd.DataFrame()
        for col in self.cols:
            curr_col: pd.Timestamp = data[col].dt
            all_time_vals: Dict[str, pd.Series] = {
                "week": curr_col.isocalendar().week,
                "weekday": curr_col.weekday,
                "day": curr_col.day,
                "hour": curr_col.hour,
                "minute": curr_col.minute,
                "second": curr_col.second,
            }

            all_tmp = {f"{col}.{time_type}": list(time_vals.values) for time_type, time_vals in all_time_vals.items() if time_vals.nunique() > 1}
            all_tmp_df = pd.DataFrame(all_tmp)
            time_data = pd.concat([time_data, all_tmp_df], axis=1)
            result_cols = list(all_tmp_df.columns)
            self.pre2post = {**self.pre2post, col: result_cols}
            self.post2pre = {**self.post2pre, **{rcol: col for rcol in result_cols}}
        # test = self.col2times.keys
        data = data.drop(self.cols, axis=1)
        data = pd.concat([data, time_data], axis=1)
        return data, {}

    def _extract_time(self, time_type: str, time_vals: pd.Series):
        if time_vals.nunique() > 1:
            new_col_name = f"ts.{time_type}"
            new_time_col = {new_col_name: time_vals.values}
        return new_time_col

    def revert(self, data: pd.DataFrame, **kwargs):
        time_data = pd.DataFrame()
        for col_timestamp in self.pre2post.keys:
            all_cols_for_col_timestamp = [col for col in data.columns if col_timestamp in col]
            t_data = data[all_cols_for_col_timestamp].apply(pd.to_datetime)
            time_data = time_data.join(t_data)
        return time_data, {}


class StandardOperations(ABC):
    @staticmethod
    def drop_cols(cols: List[str] = None):
        def process(data: pd.DataFrame):
            # result = result.copy()
            if not cols:
                return data, {"dropped_cols": [], "remaining_cols": list(data.columns)}
            # removed_cols = set(data.columns) - set(new_data.columns)
            data = data.drop(cols, axis=1)
            return data, {"dropped_cols": cols, "remaining_cols": list(data.columns)}

        def revert(data: pd.DataFrame):
            # result = result.copy()
            return data, {}

        return process, revert

    @staticmethod
    def set_index(col_case_id: str = None):
        def process(data: pd.DataFrame):
            if col_case_id is None:
                return data
            return data.set_index(col_case_id), {"index_col": col_case_id}

        def revert(data: pd.DataFrame):
            # result = result.copy()
            return data.reset_index(), {}

        return process, revert

    @staticmethod
    def extract_time(cols: List[str] = None):
        def process(data: pd.DataFrame):
            time_data = pd.DataFrame()
            time_vals: pd.Series = None
            for col_timestamp in cols:
                tmp_df = pd.DataFrame()
                time_vals = data[col_timestamp].dt.isocalendar().week
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.week"] = time_vals

                time_vals = data[col_timestamp].dt.weekday
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.weekday"] = time_vals

                time_vals = data[col_timestamp].dt.day
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.day"] = time_vals

                time_vals = data[col_timestamp].dt.hour
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.hour"] = time_vals

                time_vals = data[col_timestamp].dt.minute
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.minute"] = time_vals

                time_vals = data[col_timestamp].dt.second
                if time_vals.nunique() > 1:
                    tmp_df[f"{col_timestamp}.ts.second"] = time_vals

                time_data = time_data.join(tmp_df)
            data = data.drop(cols, axis=1)
            data = data.join(time_data)
            return data, {"time_cols": cols}

        def revert(data: pd.DataFrame):
            # result = result.copy()
            time_data = pd.DataFrame()
            for col_timestamp in cols:
                all_cols_for_col_timestamp = [col for col in data.columns if col_timestamp in col]
                t_data = data[all_cols_for_col_timestamp].apply(pd.to_datetime)
                time_data = time_data.join(t_data)
            return time_data, {"time_cols": cols}

        return process, revert


class SetIndexOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols = cols if isinstance(cols, list) else [cols]
        self.pre2post = {}
        self.post2pre = None
        
    def digest(self, data: pd.DataFrame, **kwargs):
        new_data = data.set_index(self.cols)
        self.pre2post = {col: col in new_data.index for col in self.cols}
        return new_data, {}
    
    def revert(self, data: pd.DataFrame, **kwargs):
        return data.reset_index(), {}


class ProcessingPipeline():
    def __init__(self, ):
        self.root = None

    def set_root(self, root: Operation):
        self.root = root
        return self

    def transform(self, data:pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        data, info = self.root.forward(data, **kwargs)
        return data, info

    def __getitem__(self, key) -> Operation:
        curr_pos = self.root
        visited: List[Operation] = []
        queue: List[Operation] = []
        visited.append(self.root)
        queue.append(self.root)
        while len(queue):
            curr_pos = queue.pop(0)
            for child in curr_pos._next:
                if child not in visited:
                    if child.name == key:
                        return child

                    visited.append(child)
                    queue.append(child)
        return None
