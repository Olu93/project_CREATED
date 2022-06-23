from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union
if TYPE_CHECKING:
    from thesis_readers.readers.AbstractProcessLogReader import ImportantCols

from abc import ABC, abstractmethod
from collections import UserList
from numbers import Number
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


class ColStats(BetterDict):
    def __init__(self, important_cols: ImportantCols, min_diversity=0.0, max_diversity=0.8, max_similarity=0.6, max_missing=0.75, **kwargs):
        super().__init__({
            'config': {
                "min_diversity": min_diversity,
                "max_diversity": max_diversity,
                "max_similarity": max_similarity,
                "max_missing": max_missing,
                "important_cols": important_cols
            },
            'cols': kwargs.get('cols')
        })

    def compute(self, data: pd.DataFrame):
        icols = self.get('config', {}).get('important_cols')
        min_div = self.get('config', {}).get('min_diversity')
        max_div = self.get('config', {}).get('max_diversity')
        max_sim = self.get('config', {}).get('max_similarity')
        max_mis = self.get('config', {}).get('max_missing')

        col_statistics = ColStats._gather_column_statsitics(data, icols)
        col_statistics = {col: {**stats, "uselessness": ColStats._get_uselessness(stats, icols, min_div, max_div, max_sim, max_mis)} for col, stats in col_statistics.items()}
        col_statistics = {col: {**stats, "is_useless": ColStats._decide_uselessness(stats)} for col, stats in col_statistics.items()}
        return ColStats(important_cols=icols, cols=col_statistics)
    
    @property
    def cols(self):
        return self.get('cols')

    @property
    def important_cols(self):
        return self.get('config').get('important_cols')

    @staticmethod
    def _gather_column_statsitics(df: pd.DataFrame, important_cols: ImportantCols):
        full_len = len(df)
        # if df.index.name:
        num_traces = df.index.nunique(False) if important_cols.col_case_id == df.index.name else df[important_cols.col_case_id].nunique(False)
        
        results = {
            col: {
                'name': col,
                'diversity': df[col].nunique(False) / full_len if df[col].nunique(False) > 1 else 0,  # Special case of just one unique
                'dtype': str(df[col].dtype),
                'missing_ratio': df[col].isna().sum() / full_len,
                'intracase_similarity': 1 - (np.abs(df[col].nunique(False) - num_traces) / np.max([df[col].nunique(False), num_traces])),
                '_num_unique': df[col].nunique(False),
                'is_numeric': bool(ColStats._is_numeric(df[col])),
                'is_binary': bool(ColStats._is_binary(df[col])),
                'is_categorical': bool(ColStats._is_categorical(df[col])),
                'is_timestamp': bool(ColStats._is_timestamp(df[col])),
                'is_singular': bool(ColStats._is_singular(df[col])),
                'is_col_case_id': important_cols.col_case_id == col,
                'is_col_timestamp': important_cols.col_timestamp == col,
                'is_col_outcome': important_cols.col_outcome == col,
                'is_col_activity_id': important_cols.col_activity_id == col,
                'is_important': col in important_cols,
                '_num_rows': full_len,
                '_num_traces': num_traces,
            }
            for col in df.columns
        }
        return results

    @staticmethod
    def _get_uselessness(stats, important_cols, min_diversity, max_diversity, max_similarity, max_missing):
        is_singular = stats.get("diversity") == 0
        is_diverse = stats.get("diversity") > min_diversity
        is_naturally_diverse = (stats.get('is_numeric') or stats.get("is_timestamp"))
        is_not_diverse_enough = ((not is_diverse) & (not is_naturally_diverse))
        is_unique_to_case = (stats.get("intracase_similarity") > max_similarity)
        is_missing_too_many = (stats.get("missing_ratio") > max_missing)
        return {
            "is_singular": is_singular,
            "is_not_diverse_enough": is_not_diverse_enough,
            "is_unique_to_case": is_unique_to_case,
            "is_missing_too_many": is_missing_too_many,
        }

    @staticmethod
    def _decide_uselessness(stats:Dict) -> bool:
        return any(stats["uselessness"].values()) and not (stats.get("is_important") or stats.get("is_timestamp"))

    @staticmethod
    def _is_categorical(series: pd.Series):
        return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and series.nunique(False) > 2

    @staticmethod
    def _is_binary(series: pd.Series):
        return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and series.nunique(False) == 2

    @staticmethod
    def _is_singular(series: pd.Series):
        return series.nunique(False) == 1

    @staticmethod
    def _is_numeric(series: pd.Series):
        return pd.api.types.is_numeric_dtype(series)

    @staticmethod
    def _is_timestamp(series: pd.Series):
        return pd.api.types.is_datetime64_any_dtype(series)


class Selector(ABC):
    def select_colstats(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return ColStats(col_stats.important_cols, **kwargs)

        return fn

    def select_static(cols: List[str], **kwargs) -> Callable:
        def fn():
            return cols

        return fn

    def select_useless(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return [col for col, stats in col_stats.cols.items() if (stats.get("is_useless"))]

        return fn

    def select_binaricals(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return [col for col, stats in col_stats.cols.items() if stats.get("is_binary") and not stats.get("is_outcome")]

        return fn

    def select_categoricals(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return [col for col, stats in col_stats.cols.items() if stats.get("is_categorical") and not (stats.get("is_col_case_id") or stats.get("is_col_activity_id"))]

        return fn

    def select_numericals(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return [col for col, stats in col_stats.cols.items() if stats.get("is_numeric")]

        return fn

    def select_timestamps(col_stats: ColStats, **kwargs) -> Callable:
        def fn():
            return [col for col, stats in col_stats.cols.items() if stats.get("is_timestamp")]

        return fn


class Operation(ABC):
    def __init__(self, name: str = "default", digest_fn: Callable = None, revert_fn: Callable = None, **kwargs: BetterDict):
        self.pre2post = {}
        self.post2pre = {}
        self._next: List[Operation] = []
        self._prev: List[Operation] = []
        self._selector_args: BetterDict = BetterDict(kwargs) if kwargs else BetterDict()
        self._params_r: BetterDict = BetterDict()  #
        self._digest = digest_fn
        self._revert = revert_fn
        self.name = name

    def set_params(self, **kwargs) -> Operation:
        self._selector_args = kwargs
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
        return self

    # @abstractmethod
    def forward(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, BetterDict]:
        if self.name == 'label_encode':
            print("Stop")
        if self.name == 'time_extraction':
            print("Stop")
        digest_args = {**self._selector_args, **kwargs}
        post_data, pass_over_args = self.digest(data, **digest_args)

        params_r_collector = BetterDict(pass_over_args)
        for child in self._next:
            post_data, pass_over_args = child.forward(post_data, **pass_over_args)
            params_r_collector[child.name] = pass_over_args

        self._result = post_data.copy()
        self._params_r = params_r_collector.copy()
        return self._result, self._params_r

    # @abstractmethod
    def backward(self, data: pd.DataFrame, **kwargs):
        self._selector_args = kwargs
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


class ComputeColStatsOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)

    def digest(self, data: pd.DataFrame, **kwargs):
        self.stats = self._digest(**kwargs)()
        return data, {'col_stats': self.stats.compute(data)}

    def revert(self, data: pd.DataFrame, **kwargs):
        return data, {'col_stats': self.stats.compute(data)}


class DropOperation(IrreversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        new_data = data.drop(self.cols, axis=1)
        self.pre2post = {col: col in new_data.columns for col in self.cols}
        return new_data, {'col_stats': kwargs.get('col_stats')}


class DropUselessOperation(DropOperation):
    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        new_data = data.drop(self.cols, axis=1)
        self.pre2post = {col: col in new_data.columns for col in self.cols}
        return new_data, {'col_stats': kwargs.get('col_stats')}


class BinaryEncodeOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        if not len(self.cols):
            return data, {}
        encoder = preprocessing.OneHotEncoder(drop='if_binary', sparse=False)
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data[self.cols] = new_data
        self.pre2post = {col: [col] for col in self.cols}
        self.post2pre = {col: col for col in self.cols}
        return data, {'col_stats': kwargs.get('col_stats')}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.post2pre.keys
        data[keys] = self.encoder.inverse_transform(data[keys])
        return data, {}


class CategoryEncodeOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        if not len(self.cols):
            return data, {'col_stats': kwargs.get('col_stats')}
        encoder = ce.BaseNEncoder(return_df=True, drop_invariant=True, base=2)
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data = pd.concat([data, new_data], axis=1)
        for col in self.cols:
            result_cols = [ft for ft in encoder.get_feature_names() if col in ft]
            self.pre2post = {**self.pre2post, col: result_cols}
            self.post2pre = {**self.post2pre, **{rcol: col for rcol in result_cols}}
        return data, {'col_stats': kwargs.get('col_stats')}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = np.unique(list(self.pre2post.keys))
        for k in keys:
            mappings = self.pre2post[k]
            values = [m.value for m in mappings]
            data[k] = self.encoder.inverse_transform(data[values]).drop(values, axis=1)
        return data, {}


class NumericalEncodeOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.encoder = None

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        if not len(self.cols):
            return data, {'col_stats': kwargs.get('col_stats')}
        encoder = preprocessing.StandardScaler()
        self.encoder = encoder
        new_data = encoder.fit_transform(data[self.cols])
        data = data.drop(self.cols, axis=1)
        data[self.cols] = new_data
        self.pre2post = {col: [col] for col in self.cols}
        self.post2pre = {col: col for col in self.cols}
        return data, {'col_stats': kwargs.get('col_stats')}

    def revert(self, data: pd.DataFrame, **kwargs):
        keys = self.post2pre.keys
        data[keys] = self.encoder.inverse_transform(data[keys])
        return data, {}


class TimeExtractOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
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

            all_tmp = {f"{col}/{time_type}": list(time_vals.values) for time_type, time_vals in all_time_vals.items() if time_vals.nunique() > 1}
            all_tmp_df = pd.DataFrame(all_tmp)
            time_data = pd.concat([time_data, all_tmp_df], axis=1)
            result_cols = list(all_tmp_df.columns)
            self.pre2post = {**self.pre2post, col: result_cols}
            self.post2pre = {**self.post2pre, **{rcol: col for rcol in result_cols}}
        # test = self.col2times.keys
        data = data.drop(self.cols, axis=1)
        data = pd.concat([data, time_data.set_index(data.index)], axis=1)
        return data, {"new_cols": list(time_data.columns), 'col_stats': kwargs.get('col_stats')}

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


class SetIndexOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)()
        new_data = data.set_index(self.cols)
        self.pre2post = {col: col in new_data.index for col in self.cols}
        return new_data, {'col_stats': kwargs.get('col_stats')}

    def revert(self, data: pd.DataFrame, **kwargs):
        return data.reset_index(), {}


class ProcessingPipeline():
    def __init__(self, root: Operation = None):
        self.root: Operation = root
        self._data: pd.DataFrame = None
        self._info: Dict = None

    def set_root(self, root: Operation):
        self.root = root
        return self

    def fit(self, data: pd.DataFrame, **kwargs) -> ProcessingPipeline:
        self._data, self._info = self.root.forward(data, **kwargs)
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def info(self) -> Dict:
        return self._info

    def collect_as_list(self) -> List[Operation]:
        curr_pos = self.root
        visited: List[Operation] = []
        queue: List[Operation] = []
        visited.append(self.root)
        queue.append(self.root)
        while len(queue):
            curr_pos = queue.pop(0)
            for child in curr_pos._next:
                if child not in visited:
                    visited.append(child)
                    queue.append(child)
        return visited

    def collect_as_dict(self) -> Dict[str, Operation]:
        all_operations = self.collect_as_list()
        return {op.name: op for op in all_operations}

    @property
    def mapping(self) -> Dict:
        all_operations = self.collect_as_list()
        return {op.name: op.pre2post for op in all_operations if hasattr(op, 'pre2post')}

    @property
    def reverse_mapping(self) -> Dict:
        all_operations = self.collect_as_list()
        return {op.name: op.post2pre for op in all_operations if hasattr(op, 'post2pre')}

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


class LabelEncodeOperation(ReversableOperation):
    def __init__(self, **kwargs: BetterDict):
        super().__init__(**kwargs)
        self.col2pp: Dict[str, preprocessing.LabelEncoder] = {}

    def digest(self, data: pd.DataFrame, **kwargs):
        self.cols = self._digest(**kwargs)
        for col in self.cols:
            if col not in data:
                continue
            self.col2pp[col] = preprocessing.LabelEncoder().fit(data[col])
            data[col] = self.col2pp[col].transform(data[col])
            self.pre2post[col] = [col]
            self.post2pre[col] = col
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
