import time
import random
from enum import IntEnum, auto, Enum
from typing import Counter, Dict, Iterable, Iterator, List, Tuple, Union
import pathlib
from matplotlib import pyplot as plt
import pandas as pd
import pm4py
from IPython.display import display
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.util import constants
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.visualization.petrinet import visualizer as petrinet_visualization
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from sklearn import preprocessing
import itertools as it
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

from thesis_readers.misc.constants import DATA_FOLDER, DATA_FOLDER_PREPROCESSED, DATA_FOLDER_VISUALIZATION

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG


class TaskModes(Enum):
    SIMPLE = auto()
    EXTENSIVE = auto()
    EXTENSIVE_RANDOM = auto()
    FINAL_OUTCOME = auto()
    ENCODER_DECODER = auto()


class DatasetModes(IntEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class ShapeModes(IntEnum):
    FULL = auto()
    FULL_SEP = auto()
    EVENT_ONLY = auto()
    EVENT_TIME = auto()


# class TargetModes(IntEnum):
#     FULL = auto()
#     FULL_SEP = auto()
#     EVENT_ONLY = auto()
#     EVENT_TIME = auto()


class AbstractProcessLogReader():
    """DatasetBuilder for my_dataset dataset."""

    log = None
    log_path: str = None
    _original_data: pd.DataFrame = None
    data: pd.DataFrame = None
    debug: bool = False
    col_case_id: str = None
    col_activity_id: str = None
    _vocab: dict = None
    mode: TaskModes = TaskModes.SIMPLE
    padding_token: str = "<P>"
    end_token: str = "<E>"
    start_token: str = "<S>"
    transform = None
    time_stats = {}

    def __init__(self,
                 log_path: str,
                 csv_path: str,
                 col_case_id: str = 'case:concept:name',
                 col_event_id: str = 'concept:name',
                 col_timestamp: str = 'time:timestamp',
                 debug=False,
                 mode: TaskModes = TaskModes.SIMPLE,
                 max_tokens: int = None,
                 **kwargs) -> None:
        super(AbstractProcessLogReader, self).__init__(**kwargs)
        self.vocab_len = None
        self.debug = debug
        self.mode = mode
        self.log_path = pathlib.Path(log_path)
        self.csv_path = pathlib.Path(csv_path)
        self.col_case_id = col_case_id
        self.col_activity_id = col_event_id
        self.col_timestamp = col_timestamp
        self.preprocessors = {}

    def init_log(self, save=False):
        self.log = pm4py.read_xes(self.log_path.as_posix())
        if self.debug:
            print(self.log[1])  #prints the first event of the first trace of the given log
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if save:
            self._original_data.to_csv(self.csv_path, index=False)
        return self

    def init_data(self):
        start_time = time.time()

        self._original_data = self._original_data if self._original_data is not None else pd.read_csv(self.csv_path)
        self._original_data = dataframe_utils.convert_timestamp_columns_in_df(
            self._original_data,
            timest_columns=[self.col_timestamp],
        )
        if self.debug:
            display(self._original_data.head())

        parameters = {TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.col_case_id}
        self.log = self.log if self.log is not None else log_converter.apply(self._original_data, parameters=parameters, variant=TO_EVENT_LOG)
        self._original_data[self.col_case_id] = self._original_data[self.col_case_id].astype('object')
        self._original_data[self.col_activity_id] = self._original_data[self.col_activity_id].astype('object')
        self.preprocess_level_general()
        self.preprocess_level_specialized()
        self.register_vocabulary()
        self.group_rows_into_traces()
        self.gather_information_about_traces()
        self.instantiate_dataset()

        self.time_stats["full_data_preprocessing_pipeline"] = time.time() - start_time
        return self

    # @staticmethod
    # def gather_grp_column_statsitics(df: pd.DataFrame):
    #     full_len = len(df)
    #     return {col: {'name': col, 'entropy': entropy(df[col].value_counts()), 'dtype': df[col].dtype, 'missing_ratio': df[col].isna().sum() / full_len} for col in df.columns}

    def preprocess_level_general(self, remove_cols=None, max_diversity_thresh=0.75, min_diversity=0.0, too_similar_thresh=0.6, missing_thresh=0.75, **kwargs):
        self.data = self.original_data
        if remove_cols:
            self.data = self.data.drop(remove_cols, axis=1)
        col_statistics = self._gather_column_statsitics(self.data.select_dtypes('object'))
        col_statistics = {
            col: dict(stats, is_useless=self._is_useless_col(stats, min_diversity, max_diversity_thresh, too_similar_thresh, missing_thresh))
            for col, stats in col_statistics.items() if col not in [self.col_case_id, self.col_activity_id, self.col_timestamp]
        }
        col_statistics = {col: dict(stats, is_dropped=any(stats["is_useless"])) for col, stats in col_statistics.items()}
        cols_to_remove = [col for col, val in col_statistics.items() if val["is_dropped"]]
        self.data = self.data.drop(cols_to_remove, axis=1)

    def _is_useless_col(self, stats, min_diversity_thresh, max_diversity_thresh, similarity_ratio_thresh, missing_ratio_thresh):
        has_reasonable_diversity = (stats.get("diversity") > min_diversity_thresh and stats.get("diversity") < max_diversity_thresh)
        is_probably_unique_to_case = (stats.get("similarity_to_trace_num") > similarity_ratio_thresh)
        is_missing_too_many = stats.get("missing_ratio") > missing_ratio_thresh
        return (not has_reasonable_diversity), is_probably_unique_to_case, is_missing_too_many

    def _gather_column_statsitics(self, df: pd.DataFrame):
        full_len = len(df)
        num_traces = df[self.col_case_id].nunique(False)
        results = {
            col: {
                'name': col,
                'diversity': df[col].nunique(False) / full_len,
                'dtype': str(df[col].dtype),
                'missing_ratio': df[col].isna().sum() / full_len,
                'similarity_to_trace_num': 1 - (np.abs(df[col].nunique(False) - num_traces) / np.max([df[col].nunique(False), num_traces])),
                '_num_unique': df[col].nunique(False),
                '_num_rows': full_len,
                '_num_traces': num_traces,
            }
            for col in df.columns
        }
        return results

    def preprocess_level_specialized(self, **kwargs):
        cols = kwargs.get('cols', self.data.columns)
        # Prepare remaining columns
        for col in cols:
            if col in [self.col_case_id, self.col_activity_id, self.col_timestamp]:
                continue
            if pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            self.preprocessors[col] = preprocessing.LabelEncoder().fit(self.data[col])
            self.data[col] = self.preprocessors[col].transform(self.data[col])

        # Prepare timestamp
        self.data["month"] = self.data[self.col_timestamp].dt.month
        self.data["week"] = self.data[self.col_timestamp].dt.isocalendar().week
        self.data["weekday"] = self.data[self.col_timestamp].dt.weekday
        self.data["day"] = self.data[self.col_timestamp].dt.day
        self.data["hour"] = self.data[self.col_timestamp].dt.hour
        self.data["minute"] = self.data[self.col_timestamp].dt.minute
        self.data["second"] = self.data[self.col_timestamp].dt.second
        del self.data[self.col_timestamp]
        num_encoder = StandardScaler()
        normalization_columns = [
            "month",
            "week",
            "weekday",
            "day",
            "hour",
            "minute",
            "second",
        ]
        self.preprocessors['time'] = num_encoder
        self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.set_index(self.col_case_id)

    def register_vocabulary(self):
        all_unique_tokens = list(self.data[self.col_activity_id].unique())

        self._vocab = {word: idx for idx, word in enumerate(all_unique_tokens, 1)}
        self._vocab[self.padding_token] = 0
        self._vocab[self.start_token] = len(self._vocab) + 1
        self._vocab[self.end_token] = len(self._vocab) + 1
        self.vocab_len = len(self._vocab) + 1
        self._vocab_r = {idx: word for word, idx in self._vocab.items()}

    def group_rows_into_traces(self):
        self.data = self.data.replace({self.col_activity_id: self._vocab})
        self.grouped_traces = list(self.data.groupby(by=self.col_case_id))
        self._traces = {idx: df for idx, df in self.grouped_traces}

    def gather_information_about_traces(self):
        self.length_distribution = Counter([len(tr) for tr in self._traces.values()])
        self.max_len = max(list(self.length_distribution.keys())) + 2
        self.min_len = min(list(self.length_distribution.keys())) + 2
        self.log_len = len(self._traces)
        self.feature_len = len(self.data.columns)
        self.idx_event_attribute = self.data.columns.get_loc(self.col_activity_id)
        self.feature_shapes = ((self.max_len, ), (self.max_len, self.feature_len - 1), (self.max_len, self.feature_len), (self.max_len, self.feature_len))
        self.feature_types = (tf.float32, tf.float32, tf.float32, tf.float32)

    def instantiate_dataset(self):
        print("Preprocess data")
        self.data_container = np.zeros([self.log_len, self.max_len, self.feature_len])
        loader = tqdm(self._traces.items(), total=len(self._traces))

        for idx, (case_id, df) in enumerate(loader):
            df_end = len(df) + 1
            self.data_container[idx, 1:df_end] = df.values
            self.data_container[idx, 0, self.idx_event_attribute] = self.vocab2idx[self.start_token]
            self.data_container[idx, df_end, self.idx_event_attribute] = self.vocab2idx[self.end_token]

        if self.mode == TaskModes.SIMPLE:
            self.traces = self.data_container, np.roll(self.data_container, -1, axis=1)
            self.traces[1][:, -1] = 0

        if self.mode == TaskModes.ENCODER_DECODER:
            self.traces = ([idx, tr[0:split], tr[split:]] for idx, tr in loader if len(tr) > 1 for split in [random.randint(1, len(tr))])

        if self.mode == TaskModes.EXTENSIVE:
            self.traces = ([tr[0:end - 1], tr[1:end]] for tr in loader for end in range(2, len(tr) + 1) if len(tr) > 1)

        if self.mode == TaskModes.EXTENSIVE_RANDOM:
            tmp_traces = [tr[random.randint(0, len(tr) - 1):] for tr in loader for sample in self._heuristic_bounded_sample_size(tr) if len(tr) > 1]
            self.traces = [tr[:random.randint(2, len(tr))] for tr in tqdm(tmp_traces, desc="random-samples") if len(tr) > 1]

        if self.mode == TaskModes.FINAL_OUTCOME:
            self.traces = ([idx, tr[0:], tr[-2] * (len(tr) - 1)] for idx, tr in loader if len(tr) > 1)

        self.traces, self.targets = self.traces

        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)

        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")

    def viz_dfg(self, bg_color="transparent", save=False):
        start_time = time.time()
        dfg = dfg_discovery.apply(self.log)
        gviz = dfg_visualization.apply(dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        gviz.graph_attr["bgcolor"] = bg_color
        self.time_stats["visualize_dfg"] = time.time() - start_time
        if save:
            return dfg_visualization.save(gviz, DATA_FOLDER_VISUALIZATION / (type(self).__name__ + "_dfg.png"))
        return dfg_visualization.view(gviz)

    def viz_bpmn(self, bg_color="transparent", save=False):
        start_time = time.time()

        process_tree = pm4py.discover_tree_inductive(self.log)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)
        parameters = bpmn_visualizer.Variants.CLASSIC.value.Parameters
        gviz = bpmn_visualizer.apply(bpmn_model, parameters={parameters.FORMAT: 'png'})
        gviz.graph_attr["bgcolor"] = bg_color
        self.time_stats["visualize_bpmn"] = time.time() - start_time

        if save:
            return bpmn_visualizer.save(gviz, DATA_FOLDER_VISUALIZATION / (type(self).__name__ + "_bpmn.png"))
        return bpmn_visualizer.view(gviz)

    def viz_simple_process_map(self, bg_color="transparent", save=False):
        start_time = time.time()
        dfg, start_activities, end_activities = pm4py.discover_dfg(self.log)
        self.time_stats["visualize_simple_procmap"] = time.time() - start_time
        if save:
            return pm4py.save_vis_dfg(dfg, start_activities, end_activities, DATA_FOLDER_VISUALIZATION / (type(self).__name__ + "_sprocessmap.png"))
        return pm4py.view_dfg(dfg, start_activities, end_activities)

    def viz_process_map(self, bg_color="transparent", save=False):
        start_time = time.time()
        mapping = pm4py.discover_heuristics_net(self.log)
        parameters = hn_visualizer.Variants.PYDOTPLUS.value.Parameters
        gviz = hn_visualizer.apply(mapping, parameters={parameters.FORMAT: 'png'})
        # gviz.graph_attr["bgcolor"] = bg_color
        self.time_stats["visualize_procmap"] = time.time() - start_time
        if save:
            return hn_visualizer.save(gviz, DATA_FOLDER_VISUALIZATION / (type(self).__name__ + "_processmap.png"))
        return hn_visualizer.view(gviz)

    def get_data_statistics(self):
        return {
            "class_name": type(self).__name__,
            "log_size": self._log_size,
            "min_seq_len": self._min_seq_len,
            "max_seq_len": self._max_seq_len,
            "distinct_trace_ratio": self._distinct_trace_ratio,
            "num_distinct_events": self._num_distinct_events,
            "time": self.time_stats,
            "time_unit": "seconds",
            "column_stats": self._gather_column_statsitics(self.data.reset_index()),
        }

    # TODO: Change to less complicated output
    def _generate_examples(self, data_mode: int = DatasetModes.TRAIN) -> Iterator:
        """Generator of examples for each split."""
        data = None

        if DatasetModes(data_mode) == DatasetModes.TRAIN:
            data = (self.trace_train, self.target_train)
        if DatasetModes(data_mode) == DatasetModes.VAL:
            data = (self.trace_val, self.target_val)
        if DatasetModes(data_mode) == DatasetModes.TEST:
            data = (self.trace_test, self.target_test)

        res_features, res_targets = self._prepare_input_data(*data)

        for trace, target in zip(zip(*res_features), zip(*res_targets)):
            yield (trace, target)

    def _prepare_input_data(
            self,
            features: np.ndarray,
            targets: np.ndarray = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        feature_ids = list(range(self.feature_len))
        tmp_feature_ids = list(feature_ids)
        tmp_feature_ids.remove(self.idx_event_attribute)
        ft_events, ft_full, ft_rest, ft_empty = features[:, :, self.idx_event_attribute], features, features[:, :, tmp_feature_ids], np.zeros_like(features)
        res_features = (ft_events, ft_rest, ft_full, ft_empty)

        if targets is not None:
            tt_events, tt_full, tt_rest, tt_empty = targets[:, :, self.idx_event_attribute], targets, targets[:, :, tmp_feature_ids], np.zeros_like(targets)
            res_targets = (tt_events, tt_rest, tt_full, tt_empty)
            return res_features, res_targets
        return res_features, None

    def get_dataset(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TRAIN):

        return tf.data.Dataset.from_generator(
            self._generate_examples,
            args=[data_mode],
            output_types=(self.feature_types, self.feature_types),
            output_shapes=(self.feature_shapes, self.feature_shapes),
        ).batch(batch_size)

    def prepare_input(self, features: np.ndarray, targets: np.ndarray = None):
        return tf.data.Dataset.from_tensor_slices(self._prepare_input_data(features, targets))

    def decode_matrix(self, data):
        return np.array([[self.idx2vocab[i] for i in row] for row in data])

    def _heuristic_sample_size(self, sequence):
        return range((len(sequence)**2 + len(sequence)) // 4)

    def _heuristic_bounded_sample_size(self, sequence):
        return range(min((len(sequence)**2 + len(sequence) // 4), 5))

    def _get_example_trace_subset(self, num_traces=10):
        random_starting_point = random.randint(0, self._log_size - num_traces - 1)
        df_traces = pd.DataFrame(self._traces.items()).set_index(0).sort_index()
        example = df_traces[random_starting_point:random_starting_point + num_traces]
        return [val for val in example.values]

    @property
    def original_data(self):
        return self._original_data.copy()

    @original_data.setter
    def original_data(self, data: pd.DataFrame):
        self._original_data = data

    @property
    def tokens(self) -> List[str]:
        return list(self._vocab.keys())

    @property
    def start_id(self) -> List[str]:
        return self.vocab2idx[self.start_token]

    @property
    def end_id(self) -> List[str]:
        return self.vocab2idx[self.end_token]

    @property
    def vocab2idx(self) -> List[str]:
        return self._vocab

    @property
    def idx2vocab(self) -> List[str]:
        return self._vocab_r

    @property
    def _log_size(self):
        return len(self._traces)

    @property
    def _distinct_trace_ratio(self):
        return len(set(tuple(tr) for tr in self._traces.values())) / self._log_size

    @property
    def _min_seq_len(self):
        return self.min_len - 2

    @property
    def _max_seq_len(self):
        return self.max_len - 2

    @property
    def _num_distinct_events(self):
        return len([ev for ev in self.vocab2idx.keys() if ev not in [self.padding_token]])


class CSVLogReader(AbstractProcessLogReader):
    def __init__(self, log_path: str, csv_path: str, sep=",", **kwargs) -> None:
        super().__init__(log_path, csv_path, **kwargs)
        self.sep = sep

    def init_log(self, save=False):
        self._original_data = pd.read_csv(self.log_path, sep=self.sep)
        col_mappings = {
            self.col_timestamp: "time:timestamp",
            self.col_activity_id: "concept:name",
            self.col_case_id: "case:concept:name",
        }

        self._original_data = self._original_data.rename(columns=col_mappings)
        self.col_timestamp = "time:timestamp"
        self.col_activity_id = "concept:name"
        self.col_case_id = "case:concept:name"

        self._original_data = dataframe_utils.convert_timestamp_columns_in_df(
            self._original_data,
            timest_columns=[self.col_timestamp],
        )
        parameters = {
            TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.col_case_id,
            # TO_EVENT_LOG.value.Parameters.: self.caseId,
        }
        self.log = self.log if self.log is not None else log_converter.apply(self.original_data, parameters=parameters, variant=TO_EVENT_LOG)
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if save:
            self._original_data.to_csv(self.csv_path, index=False)
        return self

    def init_data(self):
        self.col_timestamp = "time:timestamp"
        self.col_activity_id = "concept:name"
        self.col_case_id = "case:concept:name"
        return super().init_data()


def test_dataset(reader: AbstractProcessLogReader, ds_mode=DatasetModes.TRAIN, input_mode=ShapeModes.EVENT_ONLY, target_mode=ShapeModes.EVENT_ONLY):
    ds_counter = reader.get_dataset(1, ds_mode)

    for data_point in ds_counter:
        print("INPUT")
        features = data_point[0]
        print(features[0].shape)
        print(features[1].shape)
        print(features[2].shape)
        print(features[3].shape)
        print("TARGET")
        targets = data_point[1]
        print(targets[0].shape)
        print(targets[1].shape)
        print(targets[2].shape)
        print(targets[3].shape)
        break


if __name__ == '__main__':
    reader = AbstractProcessLogReader(
        log_path=DATA_FOLDER / 'dataset_bpic2020_tu_travel/RequestForPayment.xes',
        csv_path=DATA_FOLDER_PREPROCESSED / 'RequestForPayment.csv',
        mode=TaskModes.SIMPLE,
    )
    # data = data.init_log(save=0)
    reader = reader.init_data()
    print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    features, targets = reader._prepare_input_data(reader.trace_test[0:1], reader.target_test[0:1])
    print(reader.decode_matrix(features[0]))
    print(reader.get_data_statistics())