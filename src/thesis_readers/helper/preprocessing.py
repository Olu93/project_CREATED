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


class PreprocessorMappings():
    def __init__(self, mapping_list=[]) -> None:
        self._list: List[Mapping] = mapping_list

    @staticmethod
    def map1ToM(key: str, values: List) -> PreprocessorMappings:
        return PreprocessorMappings([Mapping(key, val) for val in values])

    @staticmethod
    def mapNToM(keys: List, values: List) -> PreprocessorMappings:

        tmp_list = []
        for k in keys:
            for v in values:
                tmp_list.append(Mapping(k, v))
        return PreprocessorMappings(tmp_list)

    @staticmethod
    def mapMToM(keys: str, values: List) -> PreprocessorMappings:
        return PreprocessorMappings([Mapping(k, v) for k, v in zip(keys, values)])

    @staticmethod
    def mapMTo1(keys: List, value: Any) -> PreprocessorMappings:
        return PreprocessorMappings([Mapping(key, value) for key in keys])

    def extend(self, mappings: PreprocessorMappings) -> PreprocessorMappings:
        new_list = self._list + mappings._list
        return PreprocessorMappings(new_list)

    @property
    def keys(self):
        return np.unique([mapping.key for mapping in self._list])

    def __getitem__(self, key):
        return [m.key == key for m in self._list]

    def __repr__(self):
        return repr(self._list)


class Operation(ABC):
    def __init__(self, name: str = "default", digest_fn: Callable = None, revert_fn: Callable = None, **kwargs: BetterDict):
        self.i_cols = None
        self.o_cols = None
        self._children: List[Operation] = []
        self._parents: List[Operation] = []
        self._params: BetterDict = BetterDict()
        self._params_r: BetterDict = BetterDict()  #
        self._digest = digest_fn
        self._revert = revert_fn
        self.name = name
        self._params = kwargs or BetterDict()

    def set_params(self, **kwargs) -> Operation:
        self._params = kwargs
        return self

    def append_child(self, child: Operation, **kwargs) -> Operation:
        self._children.append(child.append_parent(self))
        return self

    def chain(self, child: Operation, **kwargs) -> Operation:
        c = child.append_parent(self)
        self._children.append(c)
        return c

    def append_parent(self, parent: Operation, **kwargs) -> Operation:
        self._parents.append(parent)
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
        for child in self._children:
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


class BinaryEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols2prpr = PreprocessorMappings()
        self.prpr2cols = PreprocessorMappings()
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
        self.cols2prpr = self.cols2prpr.extend(PreprocessorMappings.mapMToM(self.cols, self.cols))
        self.prpr2cols = self.prpr2cols.extend(PreprocessorMappings.mapMToM(self.cols, self.cols))
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.prpr2cols.keys
        data[keys] = self.encoder.inverse_transform(data[keys])
        return data, {}


class CategoryEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols2prpr = PreprocessorMappings()
        self.prpr2cols = PreprocessorMappings()
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
            self.cols2prpr = self.cols2prpr.extend(PreprocessorMappings.map1ToM(col, result_cols))
            self.prpr2cols = self.prpr2cols.extend(PreprocessorMappings.mapMTo1(result_cols, col))
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = np.unique(list(self.cols2prpr.keys))
        for k in keys:
            mappings = self.cols2prpr[k]
            values = [m.value for m in mappings]
            data[k] = self.encoder.inverse_transform(data[values]).drop(values, axis=1)
        return data, {}


class NumericalEncodeOperation(ReversableOperation):
    def __init__(self, cols: List[str], **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.cols2prpr = PreprocessorMappings()
        self.prpr2cols = PreprocessorMappings()
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
        self.cols2prpr = self.cols2prpr.extend(PreprocessorMappings.mapMToM(self.cols, self.cols))
        self.prpr2cols = self.prpr2cols.extend(PreprocessorMappings.mapMToM(self.cols, self.cols))
        return data, {}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.prpr2cols.keys
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
        self.col2times = PreprocessorMappings()
        self.times2col = PreprocessorMappings()

    def digest(self, data: pd.DataFrame, **kwargs):
        time_data = pd.DataFrame()
        for col_timestamp in self.cols:
            curr_col: pd.Timestamp = data[col_timestamp].dt
            all_time_vals: Dict[str, pd.Series] = {
                "week": curr_col.isocalendar().week,
                "weekday": curr_col.weekday,
                "day": curr_col.day,
                "hour": curr_col.hour,
                "minute": curr_col.minute,
                "second": curr_col.second,
            }

            all_tmp = {f"{col_timestamp}.{time_type}": list(time_vals.values) for time_type, time_vals in all_time_vals.items() if time_vals.nunique() > 1}
            all_tmp_df = pd.DataFrame(all_tmp)
            time_data = pd.concat([time_data, all_tmp_df], axis=1)
            new_cols = list(all_tmp_df.columns)
            self.col2times = self.col2times.extend(PreprocessorMappings.map1ToM(col_timestamp, new_cols))
            self.times2col = self.times2col.extend(PreprocessorMappings.mapMTo1(new_cols, col_timestamp))
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
        for col_timestamp in self.col2times.keys:
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


class DropOperation(IrreversableOperation):
    def forward(self, inputs, **kwargs) -> Operation:
        cols = list(kwargs.get("cols", [])) + self._params.get("cols", [])
        outputs, additional = self.drop(inputs, cols)
        self._params = additional
        self.result = outputs
        for child in self._children:
            child.forward(self.result, **self._params)
        return self

    def backward(self, inputs, **kwargs):
        return inputs

    def drop(self, inputs: pd.DataFrame, cols=None):
        result = inputs
        if not cols:
            return result, {"dropped_cols": [], "remaining_cols": list(result.columns)}
        # removed_cols = set(data.columns) - set(new_data.columns)
        result = result.drop(cols, axis=1)
        return result, {"dropped_cols": cols, "remaining_cols": list(result.columns)}


class ProcessingPipeline():
    def __init__(self, ):
        self.root = None

    def set_root(self, root: Operation):
        self.root = root
        return self
    
    def __getitem__(self, key):
        curr_pos = self.root
        visited = []
        queue = []
        visited.append(self.root)
        queue.append(self.root)
        while len(queue):
            curr_pos = queue.pop(0)
            for child in curr_pos._children:
                if child not in visited:
                    if child.name == key:
                        return child
                    
                    visited.append(child)
                    queue.append(child)
        return None 
