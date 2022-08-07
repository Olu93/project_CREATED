from __future__ import annotations
from abc import ABC, abstractmethod
from collections import UserList

import io
import itertools as it
import json
import os
import pathlib
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Counter, Dict, Iterable, Iterator, List, Sequence, Tuple, TypedDict, Union, ItemsView
if TYPE_CHECKING:
    from thesis_commons.model_commons import TensorflowModelMixin

from thesis_readers.helper.preprocessing import BinaryEncodeOperation, CategoryEncodeOperation, ColStats, ComputeColStatsOperation, DropOperation, IrreversableOperation, LabelEncodeOperation, NumericalEncodeOperation, Operation, ProcessingPipeline, ReversableOperation, Selector, SetIndexOperation, TemporalEncodeOperation, TimeExtractOperation
try:
    import cPickle as pickle
except:
    import pickle
import category_encoders as ce
import numpy as np
import pandas as pd
import pm4py
import tensorflow as tf
from IPython.display import display
# from nltk.lm import MLE, KneserNeyInterpolated
# from nltk.lm import \
#     preprocessing as \
#     nltk_preprocessing  # https://www.kaggle.com/alvations/n-gram-language-model-with-nltk
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from thesis_commons.representations import BetterDict
from thesis_commons import random
from thesis_commons.config import DEBUG_PRINT_PRECISION
from thesis_commons.constants import PATH_READERS, CDType, CDomain, CDomainMappings, CMeta
from thesis_commons.decorators import collect_time_stat
from thesis_commons.functions import (reverse_sequence_2, reverse_sequence_3, shift_seq_backward, shift_seq_forward)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER, DATA_FOLDER_PREPROCESSED, DATA_FOLDER_VISUALIZATION)

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG
# TODO: Maaaaaybe... Put into thesis_commons  -- NO IT FITS RIGHT HERE
# TODO: Checkout TimeseriesGenerator https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
# TODO: Checkout Sampling Methods    https://medium.com/deep-learning-with-keras/sampling-in-text-generation-b2f4825e1dad
# TODO: Add ColStats class with appropriate encoders

if DEBUG_PRINT_PRECISION:
    np.set_printoptions(suppress=True, edgeitems=26, linewidth=1000, precision=DEBUG_PRINT_PRECISION)
else:
    np.set_printoptions(edgeitems=26, linewidth=1000)


class ColInfo():
    # group: str
    # col: str
    # index: int
    # domain: ColDType
    # importance: ColDType
    # dtype: ColDType
    def __init__(self, group, col, index, dtype, domain, importance) -> None:
        self.is_timestamp = False
        self.group = group
        self.col = col
        self.index = index
        self.dtype = dtype
        self.domain = domain
        self.importance = importance

    def set_is_timestamp(self, is_timestamp):
        self.is_timestamp = is_timestamp
        return self

    def __repr__(self):
        return f"{vars(self)}"

    def __lt__(self, other: ColInfo):
        if not isinstance(other, ColInfo):
            raise Exception(f"Not comparable")
        return self.index < other.index


class FeatureInformation():
    def __init__(self, important_cols: ImportantCols, data_mapping: Dict, columns: pd.Index, **kwargs):
        self.columns = columns
        self.data_mapping = data_mapping
        self.important_cols = important_cols
        self.skip = self.important_cols.all
        self.col_mapping = {grp: val for grp, val in self.data_mapping.items() if (grp in CDomainMappings.ALL)}

        self.all_cols: List[ColInfo] = []
        self.all_cols.extend([
            ColInfo(self.important_cols.col_event, self.important_cols.col_event, self.columns.get_loc(self.important_cols.col_event), CDType.NON, CDomain.NON, CMeta.IMPRT),
            ColInfo(self.important_cols.col_outcome, self.important_cols.col_outcome, self.columns.get_loc(self.important_cols.col_outcome), CDType.NON, CDomain.NON, CMeta.IMPRT),
        ])
        if self.data_mapping.get(CDType.TMP):
            self.all_cols.extend([
                ColInfo(grp, col, self.columns.get_loc(col), CDType.TMP, CDomain.CONTINUOUS, CMeta.FEATS).set_is_timestamp(True)
                for grp, val in self.data_mapping.get(CDType.TMP).items() for col in val if self.important_cols.col_timestamp in col
            ])
            self.all_cols.extend([
                ColInfo(grp, col, self.columns.get_loc(col), CDType.TMP, CDomain.CONTINUOUS, CMeta.FEATS) for grp, val in self.data_mapping.get(CDType.TMP).items() for col in val
                if self.important_cols.col_timestamp not in col
            ])
        self.all_cols.extend([
            ColInfo(grp, col, self.columns.get_loc(col), CDType.BIN, CDomain.DISCRETE, CMeta.FEATS) for grp, val in self.data_mapping.get(CDType.BIN).items() for col in val
            if self.important_cols.col_timestamp not in col
        ])
        self.all_cols.extend([
            ColInfo(grp, col, self.columns.get_loc(col), CDType.CAT, CDomain.DISCRETE, CMeta.FEATS) for grp, val in self.data_mapping.get(CDType.CAT).items() for col in val
            if self.important_cols.col_timestamp not in col
        ])
        self.all_cols.extend([
            ColInfo(grp, col, self.columns.get_loc(col), CDType.NUM, CDomain.CONTINUOUS, CMeta.FEATS) for grp, val in self.data_mapping.get(CDType.NUM).items() for col in val
            if self.important_cols.col_timestamp not in col
        ])

        self.all_cols.sort()

        self._idx_dist_type = {
            vartype: {var: [self.columns.get_loc(col) for col in grp]
                      for var, grp in grps.items() if var != self.important_cols.col_outcome}
            for vartype, grps in self.col_mapping.items()
        }
        return None  # For easier breakpointing

    @property
    def col_case(self) -> str:
        return self.important_cols.col_case_id

    @property
    def col_event(self) -> str:
        return self.important_cols.col_event

    @property
    def col_timestamp(self) -> str:
        return self.important_cols.col_timestamp

    @property
    def col_outcome(self) -> str:
        return self.important_cols.col_outcome

    @property
    def idx_features(self) -> ColInfo:
        return {val.col: val.index for val in self.all_cols if val.importance == CMeta.FEATS}

    @property
    def idx_case(self) -> ColInfo:
        return None

    @property
    def idx_event(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if val.col == self.important_cols.col_event}

    @property
    def idx_outcome(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if val.col == self.important_cols.col_outcome} or None

    @property
    def idx_timestamp(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if val.is_timestamp}

    @property
    def idx_discrete(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.domain == CDomain.DISCRETE)}

    @property
    def idx_continuous(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.domain == CDomain.CONTINUOUS)}

    @property
    def idx_numericals(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.dtype == CDType.NUM)}

    @property
    def idx_binaricals(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.dtype in CDType.BIN)}

    @property
    def idx_categoricals(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.dtype in CDType.CAT)}

    @property
    def idx_temporals(self) -> Dict[str, int]:
        return {val.col: val.index for val in self.all_cols if (val.dtype in CDType.TMP)}

    @property
    def ft_len(self) -> int:
        return len(self.idx_features)


class ImportantCols(object):
    def __init__(self, col_case_id, col_activity_id, col_timestamp, col_outcome) -> None:
        self.col_case_id = col_case_id
        self.col_event = col_activity_id
        self.col_timestamp = col_timestamp
        self.col_outcome = col_outcome

    @property
    def all(self):
        return [self.col_case_id, self.col_event, self.col_timestamp, self.col_outcome]

    def __contains__(self, key):
        if key is None: return False
        if key in self.all:
            return True
        return False

    def set_col_case_id(self, col_case_id):
        self.col_case_id = col_case_id
        return self

    def set_col_activity_id(self, col_activity_id):
        self.col_event = col_activity_id
        return self

    def set_col_timestamp(self, col_timestamp):
        self.col_timestamp = col_timestamp
        return self

    def set_col_outcome(self, col_outcome):
        self.col_outcome = col_outcome
        return self

    def __repr__(self) -> str:
        return f"@{type(self).__name__}[ case > {self.col_case_id} | activity > {self.col_event} | timestamp > {self.col_timestamp} | outcome > {self.col_outcome} ]"


class AbstractProcessLogReader():
    """DatasetBuilder for my_dataset dataset."""
    pad_id = 0  # Value of empty activity
    pad_value = 0  # Value of empty feature
    pad_token: str = "<UNK>"
    end_token: str = "</s>"
    start_token: str = "<s>"
    transform = None
    time_stats = {}
    time_stats_units = ["hours", "minutes", "seconds", "milliseconds"]
    curr_reader_path: pathlib.Path = PATH_READERS / 'current_reader.txt'

    class shift_mode(IntEnum):
        NONE = 0
        NEXT = -1
        PREV = 1

    def __init__(self,
                 log_path: str,
                 csv_path: str,
                 col_case_id: str = 'case:concept:name',
                 col_event_id: str = 'concept:name',
                 col_timestamp: str = 'time:timestamp',
                 col_outcome: str = 'label',
                 na_val: str = 'missing',
                 debug=False,
                 mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE,
                 max_tokens: int = None,
                 ngram_order: int = 2,
                 **kwargs) -> None:
        super(AbstractProcessLogReader, self).__init__(**kwargs)
        self.name = type(self).__name__
        self.log = None
        self.log_path: str = None
        self.data: pd.DataFrame = None
        self.debug: bool = False
        self._vocab: dict = None
        self.mode: TaskModes = None
        self._original_data: pd.DataFrame = None
        self.debug = debug
        self.mode = mode
        self.log_path = pathlib.Path(log_path)
        self.csv_path = pathlib.Path(csv_path)
        self.important_cols: ImportantCols = ImportantCols(col_case_id, col_event_id, col_timestamp, col_outcome)
        self.preprocessors = {}
        self.ngram_order = ngram_order
        self.reader_folder: pathlib.Path = (PATH_READERS / type(self).__name__).absolute()
        self.pipeline: ProcessingPipeline = None
        self.na_val = na_val

        if not self.reader_folder.exists():
            os.mkdir(self.reader_folder)

    @collect_time_stat
    def init_log(self, save=False) -> AbstractProcessLogReader:
        self.log = pm4py.read_xes(self.log_path.as_posix())
        if self.debug:
            print(self.log[1])  #prints the first event of the first trace of the given log
        if self.debug:
            print(self.log[1][0])  #prints the first event of the first trace of the given log
        self._original_data = pm4py.convert_to_dataframe(self.log)
        if save:
            self._original_data.to_csv(self.csv_path, index=False)
        return self

    @property
    def col_case_id(self) -> str:
        return self.important_cols.col_case_id

    @property
    def col_activity_id(self) -> str:
        return self.important_cols.col_event

    @property
    def col_timestamp(self) -> str:
        return self.important_cols.col_timestamp

    @property
    def col_outcome(self) -> str:
        return self.important_cols.col_outcome

    @collect_time_stat
    def init_meta(self, skip_dynamics: bool = False) -> AbstractProcessLogReader:
        is_from_log = self._original_data is not None
        self.important_cols = self.important_cols.set_col_case_id(self.col_case_id if is_from_log else 'case:concept:name').set_col_activity_id(
            self.col_activity_id if is_from_log else 'concept:name').set_col_timestamp(self.col_timestamp if is_from_log else 'time:timestamp')

        self._original_data = self._original_data if is_from_log else pd.read_csv(self.csv_path)
        num_cols = len(self._original_data.columns)
        old_cols = set(self._original_data.columns)
        _construct = self._original_data.select_dtypes('object')
        for dtype in ['category', 'datetime64', 'datetimetz', 'timedelta64', 'number']:
            _construct = _construct.join(self._original_data.select_dtypes(dtype))
        self._original_data = _construct

        if num_cols != len(self._original_data.columns):
            new_cols = set(self._original_data.columns)
            raise Exception(f"Cols are not the same number after reorder: {old_cols-new_cols}")
        self._original_data: pd.DataFrame = dataframe_utils.convert_timestamp_columns_in_df(self._original_data,
                                                                                            # timest_columns=[self.col_timestamp],
                                                                                            )
        if self.debug:
            display(self._original_data.head())

        parameters = {TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.col_case_id}
        self.log = self.log if self.log is not None else log_converter.apply(self._original_data, parameters=parameters, variant=TO_EVENT_LOG)
        self._original_data[self.col_case_id] = self._original_data[self.col_case_id].astype('object')
        self._original_data[self.col_activity_id] = self._original_data[self.col_activity_id].astype('object')
        self._original_data = self._move_outcome_to_end(self._original_data, self.col_outcome)
        self.original_cols = list(self._original_data.columns)
        self._original_data = self._original_data.replace(self.na_val, np.nan) if self.na_val else self._original_data
        self._original_data, preprocess_kwargs = self.pre_pipeline(self._original_data)
        self.pipeline = self.preprocess(**preprocess_kwargs)
        self.data = self.pipeline.data
        self.data, _ = self.post_pipeline(self.data)
        self.data = self._move_event_to_end(self.data, self.col_activity_id)
        self.data = self._move_outcome_to_end(self.data, self.col_outcome)
        self.data_mapping = self.pipeline.mapping
        self.register_vocabulary()
        self._traces, self._grouped_traces, self.data = self.group_rows_into_traces(self.data)
        self.gather_information_about_traces()
        if not skip_dynamics:
            # self.compute_trace_dynamics()
            pass
        if self.mode is not None:
            self.instantiate_dataset(self.mode)

        return self

    def _move_outcome_to_end(self, data: pd.DataFrame, col_outcome):
        cols = list(data.columns.values)
        cols.pop(cols.index(col_outcome))
        return data[cols + [col_outcome]]

    def _move_event_to_end(self, data: pd.DataFrame, col_event):
        cols = list(data.columns.values)
        cols.pop(cols.index(col_event))
        return data[cols + [col_event]]

    # @staticmethod
    # def gather_grp_column_statsitics(df: pd.DataFrame):
    #     full_len = len(df)
    #     return {col: {'name': col, 'entropy': entropy(df[col].value_counts()), 'dtype': df[col].dtype, 'missing_ratio': df[col].isna().sum() / full_len} for col in df.columns}

    def pre_pipeline(self, data, *args, **kwargs):
        return data, kwargs

    def post_pipeline(self, data, *args, **kwargs):
        return data, kwargs

    def construct_pipeline(self, *args, **kwargs):
        remove_cols = kwargs.get('remove_cols', [])
        # dropped_by_stats_cols = [col for col, val in col_stats.items() if (val["is_useless"]) and (col not in remove_cols)]
        # col_binary_all = [col for col, stats in col_stats.items() if stats.get("is_binary") and not stats.get("is_outcome")]
        # col_cat_all = [col for col, stats in col_stats.items() if stats.get("is_categorical") and not (stats.get("is_col_case_id") or stats.get("is_col_activity_id"))]
        # col_numeric_all = [col for col, stats in col_stats.items() if stats.get("is_numeric")]
        # col_timestamp_all = [col for col, stats in col_stats.items() if stats.get("is_timestamp")]
        # print(f"Check new representation {self.important_cols}")
        pipeline = ProcessingPipeline(ComputeColStatsOperation(name="initial_stats", digest_fn=Selector.select_colstats, col_stats=ColStats(self.important_cols)))
        op = pipeline.root
        op = op.chain(DropOperation(name="premature_drop", digest_fn=Selector.select_static, cols=remove_cols))
        op = op.chain(DropOperation(name="usability_drop", digest_fn=Selector.select_useless))
        op = op.chain(SetIndexOperation(name="set_index", digest_fn=Selector.select_static, cols=[self.important_cols.col_case_id]))
        op = op.chain(TimeExtractOperation(name="temporal_extraction", digest_fn=Selector.select_timestamps))
        op = op.append_next(TemporalEncodeOperation(name=CDType.TMP, digest_fn=Selector.select_timestamps))
        op = op.append_next(BinaryEncodeOperation(name=CDType.BIN, digest_fn=Selector.select_binaricals))
        op = op.append_next(CategoryEncodeOperation(name=CDType.CAT, digest_fn=Selector.select_categoricals))
        op = op.append_next(NumericalEncodeOperation(name=CDType.NUM, digest_fn=Selector.select_numericals))

        return pipeline

    @collect_time_stat
    def preprocess(self, *args, **kwargs):
        return self.construct_pipeline(*args, **kwargs).fit(self._original_data, *args, **kwargs)

    @collect_time_stat
    def register_vocabulary(self):
        all_unique_tokens = list(self.data[self.col_activity_id].unique())

        self._vocab = {word: idx for idx, word in enumerate(all_unique_tokens, 1)}
        self._vocab[self.pad_token] = self.pad_id
        self._vocab[self.start_token] = len(self._vocab)
        self._vocab[self.end_token] = len(self._vocab)
        all_unique_outcomes = list(self.data.get(self.col_outcome, []).unique())
        self._vocab_outcome = {word: idx for idx, word in enumerate(all_unique_outcomes)} if not self._vocab_outcome else self._vocab_outcome

    @collect_time_stat
    def group_rows_into_traces(self, data:pd.DataFrame):
        data = data.replace({self.col_activity_id: self._vocab})
        data = data.replace({self.col_outcome: self._vocab_outcome})
        _grouped_traces = list(data.groupby(by=self.col_case_id))
        _traces = {idx: df for idx, df in _grouped_traces}
        return _traces, _grouped_traces, data

    @collect_time_stat
    def gather_information_about_traces(self):
        self.length_distribution = Counter([len(tr) for tr in self._traces.values()])
        self._max_len = max(list(self.length_distribution.keys()))
        self._min_len = min(list(self.length_distribution.keys()))
        orig_length_distribution = self._original_data.groupby(self.col_case_id).count()[self.col_activity_id]
        self._orig_length_distribution = Counter(orig_length_distribution.values.tolist())
        self._orig_max_len = orig_length_distribution.max()
        self.log_len = len(self._traces)
        self.num_data_cols = len(self.data.columns)

        self.feature_info = FeatureInformation(self.important_cols, self.data_mapping, self.data.columns)
        self.outcome_distribution = Counter(
            self.data.groupby(self.col_case_id).nth(0)[self.feature_info.col_outcome].replace({
                0: self.outcome2vocab[0],
                1: self.outcome2vocab[1],
            })) if self.feature_info.col_outcome is not None else None
        # outcome_dist = self.data.groupby(self.col_case_id).nth(0)[self.col_outcome].value_counts().to_dict()
        self.feature_len = self.feature_info.ft_len

        # self.feature_shapes = ((self.max_len, ), (self.max_len, self.feature_len - 1), (self.max_len, self.feature_len), (self.max_len, self.feature_len))
        # self.feature_types = (tf.float32, tf.float32, tf.float32, tf.float32)

        # self.distinct_trace_counts[(self.start_id,)] = self.log_len
        # self.distinct_trace_weights = {tr: 1 / val for tr, val in self.distinct_trace_counts.items()}
        # self.distinct_trace_weights = {tr: sum(list(self.distinct_trace_count.values())) / val for tr, val in self.distinct_trace_count.items()}

    def get_data_statistics(self):
        outcome_dist = self.data.groupby(self.col_case_id).nth(0)[self.col_outcome].value_counts().rename(index={
            0: self.outcome2vocab[0],
            1: self.outcome2vocab[1],
        }).to_dict()
        return {
            "class_name": type(self).__name__,
            "num_cases": self.num_cases,
            "min_seq_len": self._min_len,
            "max_seq_len": self._max_len,
            "ratio_distinct_traces": self.ratio_distinct_traces,
            "num_distinct_events": self.num_distinct_events,
            "num_data_columns": self.num_data_cols,
            "num_event_features": self.feature_len,
            "length_distribution": self.length_distribution,
            "outcome_distribution": self.outcome_distribution,
            "time": dict(time_unit="seconds", **self.time_stats),
            "starting_column_stats": dict(self.pipeline.root._params_r['col_stats']['cols']),
            "orig": {
                "max_seq_len": self._orig_max_len,
                "length_distribution": self._orig_length_distribution,
            }
        }

    # @collect_time_stat
    # def compute_trace_dynamics(self):

    #     print("START computing process dynamics")
    #     self._traces_only_events = {idx: df[self.col_activity_id].values.tolist() for idx, df in self.grouped_traces}
    #     self.features_by_actvity = {activity: df.drop(self.col_activity_id, axis=1) for activity, df in list(self.data.groupby(by=self.col_activity_id))}

    #     self._traces_only_events_txt = {idx: [self.idx2vocab[i] for i in indices] for idx, indices in self._traces_only_events.items()}
    #     self.trace_counts = Counter(tuple(trace[:idx + 1]) for trace in self._traces_only_events.values() for idx in range(len(trace)))
    #     self.trace_counts_by_length = {length: Counter({trace: count for trace, count in self.trace_counts.items() if len(trace) == length}) for length in range(self.max_len)}
    #     self.trace_counts_by_length_sums = {length: sum(counter.values()) for length, counter in self.trace_counts_by_length.items()}
    #     self.trace_probs_by_length = {
    #         length: {trace: count / self.trace_counts_by_length_sums.get(length, 0)
    #                  for trace, count in counter.items()}
    #         for length, counter in self.trace_counts_by_length.items()
    #     }
    #     self.trace_counts_sum = sum(self.trace_counts.values())
    #     self.trace_probs = {trace: counts / self.trace_counts_sum for trace, counts in self.trace_counts.items()}
    #     self.trace_ngrams_hard = MLE(self.ngram_order)
    #     self.trace_ngrams_soft = KneserNeyInterpolated(self.ngram_order)
    #     self.trace_ngrams_hard.fit(*nltk_preprocessing.padded_everygram_pipeline(self.ngram_order, list(self._traces_only_events_txt.values())))
    #     self.trace_ngrams_soft.fit(*nltk_preprocessing.padded_everygram_pipeline(self.ngram_order, list(self._traces_only_events_txt.values())))
    #     print("END computing process dynamics")

    @collect_time_stat
    def instantiate_dataset(self, mode: TaskModes = None, add_start: bool = None, add_end: bool = None):
        # TODO: Add option to mirror train and target
        # TODO: Add option to add boundary tags
        print(f"================ Preprocess data {self.name} ================")
        self.mode = mode or self.mode or TaskModes.NEXT_OUTCOME
        self.data_container = self._put_data_to_container(self._traces)
        # self.data_container[idx, -1, self.idx_event_attribute] = self.vocab2idx[self.end_token]
        # self.data_container[idx, -df_end - 1, self.idx_event_attribute] = self.vocab2idx[self.start_token]

        initial_data = np.array(self.data_container)
        features_container, target_container = self._preprocess_containers(self.mode, add_start, add_end, initial_data)
        self.traces_preprocessed = features_container, target_container
        self.traces, self.targets = self.traces_preprocessed
        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
        print(f"All: {len(self.traces)} datapoints")
        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")
        return self

    def _preprocess_containers(self, mode, add_start, add_end, initial_data):
        if mode == TaskModes.NEXT_EVENT_EXTENSIVE:
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, AbstractProcessLogReader.shift_mode.NEXT)
            target_container = all_next_activities

        if mode == TaskModes.NEXT_EVENT:
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, AbstractProcessLogReader.shift_mode.NEXT)
            tmp = [(ft[:idx], tg[idx + 1]) for ft, tg in zip(features_container, all_next_activities) for idx in range(1, len(ft) - 1) if (tg[idx] != 0)]
            features_container = np.zeros([len(tmp), self.max_len, self.num_data_cols])
            target_container = np.zeros([len(tmp), 1], dtype=np.int32)
            for idx, (ft, tg) in enumerate(tmp):
                features_container[idx, -len(ft):] = ft
                target_container[idx] = tg

        if mode == TaskModes.NEXT_OUTCOME:  #_SUPER
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, AbstractProcessLogReader.shift_mode.NONE)

            mask = np.not_equal(features_container[:, :, self.idx_event], 0)
            target_container = all_next_activities[:, -1][:, None]
            extensive_out_come = mask * target_container
            events_per_row = np.count_nonzero(features_container[:, :, self.idx_event], axis=-1)
            starts = self.max_len - events_per_row
            ends = self.max_len * np.ones_like(starts)
            tmp = [(ft[start:start + idx + 1], tg[start + idx]) for ft, tg, start, end in zip(features_container, extensive_out_come, starts, ends) for idx in range(end - start)]

            features_container = np.zeros([len(tmp), self.max_len, self.num_data_cols])
            target_container = np.zeros([len(tmp), 1], dtype=np.int32)
            for idx, (ft, tg) in enumerate(tmp):
                features_container[idx, -len(ft):] = ft
                target_container[idx] = tg

        if mode == TaskModes.ENCODER_DECODER:
            # DEPRECATED: To complicated and not useful
            # TODO: Include extensive version of enc dec (maybe if possible)
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            events = features_container[:, :, self.idx_event]
            events = np.roll(events, 1, axis=1)
            events[:, -1] = self.end_id

            all_rows = [list(row[np.nonzero(row)]) for row in events]
            all_splits = [(idx, split) for idx, row in enumerate(all_rows) if len(row) > 1 for split in [random.integers(1, len(row) - 1)]]

            features_container = [all_rows[idx][:split] for idx, split in all_splits]
            target_container = [all_rows[idx][split:] for idx, split in all_splits]

        if self.mode == TaskModes.OUTCOME_PREDEFINED:
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            target_container = np.max(initial_data[:, :, self.idx_outcome], axis=-1)[..., None]
            self.traces_preprocessed = features_container, target_container

        if mode == TaskModes.OUTCOME:
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, AbstractProcessLogReader.shift_mode.NONE)

            target_container = all_next_activities[:, -1, None]  # ATTENTION .reshape(-1)

        if mode == TaskModes.OUTCOME_EXTENSIVE_DEPRECATED:
            # TODO: Design features like next event
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, AbstractProcessLogReader.shift_mode.NEXT)

            mask = np.not_equal(features_container[:, :, self.idx_event], 0)
            target_container = all_next_activities[:, -1][:, None]
            extensive_out_come = mask * target_container

        return features_container, target_container

    def _put_data_to_container(self, _traces:Dict):
        # data_container = np.empty([self.log_len, self.max_len, self.num_data_cols])
        # data_container = np.ones([self.log_len, self.max_len, self.num_data_cols]) * -42
         
        data_container = np.ones([len(_traces), self.max_len, self.num_data_cols]) * self.pad_value
        # data_container[:, :, self.idx_event_attribute] = self.pad_id
        # data_container[:, :, self.idx_features] = self.pad_value
        # data_container[:] = np.nan
        loader = tqdm(_traces.items(), total=len(_traces))
        for idx, (case_id, df) in enumerate(loader):
            trace_len = len(df)
            start = self.max_len - trace_len - 1
            end = self.max_len - 1
            data_container[idx, start:end] = df.values
        return data_container

    def _add_boundary_tag(self, data_container, start_tag=False, end_tag=False):
        results = np.array(data_container)
        if ((not start_tag) and (not end_tag)):
            results = data_container
            del self._vocab[self.end_token]
            del self._vocab[self.start_token]
            return results
        if (start_tag and (not end_tag)):
            results[:, -1, self.idx_event] = self.end_id
            results = reverse_sequence_2(results)
            results = np.roll(results, 1, axis=1)
            results[:, 0, self.idx_event] = self.start_id
            results = reverse_sequence_2(results)
            results[:, -1, self.idx_event] = 0
            results = np.roll(results, 1, axis=1)
            del self._vocab[self.end_token]
            return results
        if ((not start_tag) and end_tag):
            results[:, -1, self.idx_event] = self.end_id
            del self._vocab[self.start_token]
            self._vocab[self.end_token] = len(self._vocab)
            return results
        if (start_tag and end_tag):
            results[:, -1, self.idx_event] = self.end_id
            results = reverse_sequence_2(results)
            results = np.roll(results, 1, axis=1)
            results[:, 0, self.idx_event] = self.start_id
            results = reverse_sequence_2(results)
            return results

    def _get_events_only(self, data_container, shift=None):
        result = np.array(data_container[:, :, self.idx_event])
        if shift in [None, AbstractProcessLogReader.shift_mode.NONE]:
            return result
        if shift is AbstractProcessLogReader.shift_mode.NEXT:
            result = shift_seq_backward(result).astype(int)
        if shift is AbstractProcessLogReader.shift_mode.PREV:
            result = shift_seq_forward(result).astype(int)
        return result

    def get_event_attr_len(self, ft_mode: int = FeatureModes.FULL):
        results = self._prepare_input_data(self.trace_train[:5], None, ft_mode)
        return results[0].shape[-1] if not type(results[0]) == tuple else results[0][1].shape[-1]

    # TODO: Change to less complicated output
    def _generate_dataset(self, data_mode: int = DatasetModes.TRAIN, ft_mode: int = FeatureModes.FULL) -> Iterator:
        """Generator of examples for each split."""

        features, targets = self._choose_dataset_shard(data_mode)
        res_features, res_targets = self._prepare_input_data(features, targets, ft_mode)

        # if not weighted:
        #     # res_sample_weights[:] = 1
        #     return res_features, res_targets
        # res_sample_weights = self._compute_sample_weights(res_targets)
        # return res_features, res_targets, res_sample_weights
        return res_features, res_targets

    def _generate_dataset_from_ndarray(self, features, targets, ft_mode: int = FeatureModes.FULL) -> Iterator:
        """Generator of examples for each split."""

        res_features, res_targets = self._prepare_input_data(features, targets, ft_mode)

        # if not weighted:
        #     # res_sample_weights[:] = 1
        #     return res_features, res_targets
        # res_sample_weights = self._compute_sample_weights(res_targets)
        # return res_features, res_targets, res_sample_weights
        return res_features, res_targets

    def _attach_weight(self, dataset, weight_base=None):
        # TODO: Solve this with class so that return can be easily changed to weighted form
        res_features, res_targets = dataset
        res_sample_weights = self._compute_sample_weights(weight_base) if weight_base is not None else self._compute_sample_weights(res_targets)
        return res_features, res_targets, res_sample_weights

    def _choose_dataset_shard(self, data_mode):
        if DatasetModes(data_mode) == DatasetModes.TRAIN:
            data = (self.trace_train, self.target_train)
            print("target_train: ", np.unique(self.target_train))
        if DatasetModes(data_mode) == DatasetModes.VAL:
            data = (self.trace_val, self.target_val)
            print("target_val: ", np.unique(self.target_val))
        if DatasetModes(data_mode) == DatasetModes.TEST:
            data = (self.trace_test, self.target_test)
            print("target_test: ", np.unique(self.target_test))
        if DatasetModes(data_mode) == DatasetModes.ALL:
            data = self.traces_preprocessed
            print("target_all: ", np.unique(self.target_test))
        return data

    def _prepare_input_data(
        self,
        features: np.ndarray,
        targets: np.ndarray = None,
        ft_mode: int = FeatureModes.FULL,
    ) -> tuple:
        res_features = None
        res_targets = None
        res_sample_weights = None
        # TODO: Delete everything as mode will be handled by model. Also use Population class instead.
        if ft_mode == FeatureModes.ENCODER_DECODER:
            res_features = features
        if ft_mode == FeatureModes.EVENT:
            res_features = (features[:, :, self.idx_event], np.zeros_like(features[:, :, self.idx_features]))
        if ft_mode == FeatureModes.FEATURE:
            res_features = (np.zeros_like(features[:, :, self.idx_event]), features[:, :, self.idx_features])
        if ft_mode == FeatureModes.TIME:
            res_features = (features[:, :, self.idx_event], features[:, :, self.idx_timestamp])
        if ft_mode == FeatureModes.FULL:
            res_features = (features[:, :, self.idx_event], features[:, :, self.idx_features])

        if not ft_mode == FeatureModes.ENCODER_DECODER:
            self.feature_len = res_features.shape[-1] if not type(res_features) == tuple else res_features[1].shape[-1]
        if targets is not None:

            # res_sample_weights = np.ones_like(res_sample_weights)
            # res_sample_weights = res_sample_weights if res_sample_weights.shape[1] != 1 else res_sample_weights.flatten()

            res_targets = targets
            # res_targets = res_targets if res_targets.shape[1] != 1 else res_targets.flatten()

            return res_features, res_targets
        return res_features, None

    def _compute_sample_weights(self, targets):
        # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
        # TODO: Default return weight might be tweaked to 1/len(features) or 1
        target_counts = Counter(tuple(row) for row in targets)
        sum_vals = sum(list(target_counts.values()))
        target_weigts = {k: sum_vals / v for k, v in target_counts.items()}
        weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
        return weighting

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     target_counts = Counter(tuple(row) for row in targets)
    #     target_weigts = {k: 1/v for k, v in target_counts.items()}
    #     weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
    #     return weighting/weighting.sum()

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     target_counts = Counter(tuple(row) for row in targets)
    #     target_weigts = {k: 1/v for k, v in target_counts.items()}
    #     weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
    #     return weighting

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     weighting = np.array([1 for row in targets])[:, None]
    #     return weighting

    def get_dataset(self, ds_mode: DatasetModes, ft_mode: FeatureModes, batch_size=1, num_data: int = None):
        res_data, res_targets = self._generate_dataset(ds_mode, ft_mode)
        dataset = tf.data.Dataset.from_tensor_slices((res_data, res_targets)).batch(batch_size)
        dataset = dataset.take(num_data) if num_data else dataset

        return dataset

    def get_test_dataset(self, ds_mode: DatasetModes, ft_mode: FeatureModes):
        res_data, res_targets = self._generate_dataset(ds_mode, ft_mode)
        return res_data, res_targets

    def get_dataset_generative(self, ds_mode: DatasetModes, ft_mode: FeatureModes, batch_size=1, num_data: int = None, flipped_input=False, flipped_output=False):
        # TODO: Maybe return Population object instead also rename population to Cases
        data_input, data_target = self._generate_dataset(ds_mode, ft_mode)

        results = (reverse_sequence_2(data_input[0]), reverse_sequence_2(data_input[1])) if flipped_input else (data_input[0], data_input[1])

        self.feature_len = data_input[1].shape[-1]
        dataset = tf.data.Dataset.from_tensor_slices((results, results)).batch(batch_size, drop_remainder=True)
        dataset = dataset.take(num_data) if num_data else dataset

        return dataset

    # def get_distribution(self, ds_mode: DatasetModes, ft_mode: FeatureModes):
    #     res_data, res_targets = self._generate_dataset(ds_mode, ft_mode)
    #     DataDistribution

    def get_dataset_example(self, indices=[0, 1], ds_mode: DatasetModes = DatasetModes.TRAIN, ft_mode: FeatureModes = FeatureModes.FULL):
        trace, target = self._generate_dataset(ds_mode, ft_mode)
        subset = trace, target
        if indices is not None:
            subset = (trace[0][indices], trace[1][indices]), target[indices]
        return subset

    def get_dataset_with_indices(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TEST, ft_mode: FeatureModes = FeatureModes.FULL):
        collector = []
        dataset = None
        # dataset = self.get_dataset(1, data_mode, ft_mode)
        trace, target = self._choose_dataset_shard(data_mode)
        res_features, res_targets = self._prepare_input_data(trace, target, ft_mode)
        res_features, res_targets, res_sample_weights = self._attach_weight((res_features, res_targets), res_targets)
        res_indices = trace[:, :, self.idx_event].astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((res_indices, res_features, res_targets, res_sample_weights))
        # for indices, features, target, weights in dataset:
        #     instance = ((indices, features) if type(features) is not tuple else (indices, ) + features) + (target, )
        #     collector.append(instance)
        # full_dataset = tf.data.Dataset.from_tensor_slices(tuple(collector)).batch(batch_size)
        return dataset

    def gather_full_dataset(self, dataset: tf.data.Dataset, is_generative=False):
        collector = []
        instance_x_ev = []
        instance_x_ft = []
        instance_y_ev = []
        instance_y_ft = []
        if not is_generative:
            for data_point in dataset:
                X, Y = data_point
                instance_x_ev.extend(X[0].numpy())
                instance_x_ft.extend(X[1].numpy())
                instance_y_ev.extend(Y[0].numpy())
                instance_y_ft.extend(Y[1].numpy())
            collector = [instance_x_ev, instance_x_ft, instance_y_ev, instance_y_ft]
            x_ev, x_ft, y_ev, y_ft = collector
            stacked_x_ev, stacked_x_ft, stacked_y_ev, stacked_y_ft = np.stack(x_ev), np.stack(x_ft), np.stack(y_ev), np.stack(y_ft)
        if is_generative:
            for data_point in dataset:
                X = data_point
                instance_x_ev.extend(X[0].numpy())
                instance_x_ft.extend(X[1].numpy())
                instance_y_ev.extend(X[0].numpy())
                instance_y_ft.extend(X[1].numpy())
            collector = [instance_x_ev, instance_x_ft, instance_y_ev, instance_y_ft]
            x_ev, x_ft, y_ev, y_ft = collector
            stacked_x_ev, stacked_x_ft, stacked_y_ev, stacked_y_ft = np.stack(x_ev), np.stack(x_ft), np.stack(y_ev), np.stack(y_ft)
        # Until -2 to ignore sample weight
        return stacked_x_ev, stacked_x_ft, stacked_y_ev, stacked_y_ft

    def prepare_input(self, features: np.ndarray, targets: np.ndarray = None):
        return tf.data.Dataset.from_tensor_slices(self._prepare_input_data(features, targets))

    def decode_matrix(self, data):
        return np.array([[self.idx2vocab[i] for i in row] for row in data])

    def decode_matrix_str(self, data):
        return np.array([[str(self.idx2vocab[i]) for i in row] for row in data])

    def _heuristic_sample_size(self, sequence):
        return range((len(sequence)**2 + len(sequence)) // 4)

    def _heuristic_bounded_sample_size(self, sequence):
        return range(min((len(sequence)**2 + len(sequence) // 4), 5))

    def _get_example_trace_subset(self, num_traces=10):
        random_starting_point = random.integers(0, self.num_cases - num_traces - 1)
        df_traces = pd.DataFrame(self._traces.items()).set_index(0).sort_index()
        example = df_traces[random_starting_point:random_starting_point + num_traces]
        return [val for val in example.values]

    @collect_time_stat
    def viz_dfg(self, bg_color="transparent", save=False):
        dfg = dfg_discovery.apply(self.log)
        gviz = dfg_visualization.apply(dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        gviz.graph_attr["bgcolor"] = bg_color
        if save:
            return dfg_visualization.save(gviz, self.reader_folder / "dfg.png")
        return dfg_visualization.view(gviz)

    @collect_time_stat
    def viz_bpmn(self, bg_color="transparent", save=False):
        process_tree = pm4py.discover_tree_inductive(self.log)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)
        parameters = bpmn_visualizer.Variants.CLASSIC.value.Parameters
        gviz = bpmn_visualizer.apply(bpmn_model, parameters={parameters.FORMAT: 'png'})
        gviz.graph_attr["bgcolor"] = bg_color
        if save:
            return bpmn_visualizer.save(gviz, self.reader_folder / "bpmn.png")
        return bpmn_visualizer.view(gviz)

    @collect_time_stat
    def viz_simple_process_map(self, bg_color="transparent", save=False):
        dfg, start_activities, end_activities = pm4py.discover_dfg(self.log)
        if save:
            return pm4py.save_vis_dfg(dfg, start_activities, end_activities, self.reader_folder / "simple_processmap.png")
        return pm4py.view_dfg(dfg, start_activities, end_activities)

    @collect_time_stat
    def viz_process_map(self, bg_color="transparent", save=False):
        mapping = pm4py.discover_heuristics_net(self.log)
        parameters = hn_visualizer.Variants.PYDOTPLUS.value.Parameters
        gviz = hn_visualizer.apply(mapping, parameters={parameters.FORMAT: 'png'})
        # gviz.graph_attr["bgcolor"] = bg_color
        if save:
            return hn_visualizer.save(gviz, self.reader_folder / "processmap.png")
        return hn_visualizer.view(gviz)

    def save_all_viz(self, skip_bpmn=False, skip_dfg=False, skip_spm=False, skip_pm=True):
        if not skip_bpmn:
            self.viz_bpmn(save=True)
        if not skip_dfg:
            self.viz_dfg(save=True)
        if not skip_spm:
            self.viz_simple_process_map(save=True)
        if not skip_pm:
            self.viz_process_map(save=True)

    @property
    def idx_event(self):
        return list(self.feature_info.idx_event.values())[0]

    @property
    def idx_timestamp(self):
        return list(self.feature_info.idx_timestamp.values())

    @property
    def idx_features(self):
        return list(self.feature_info.idx_features.values())

    @property
    def idx_outcome(self):
        return list(self.feature_info.idx_outcome.values())[0]

    @property
    def idx_dist_type(self):
        return self.feature_info._idx_dist_type

    @property
    def original_data(self) -> pd.DataFrame:
        return self._original_data.copy()

    @original_data.setter
    def original_data(self, data: pd.DataFrame):
        self._original_data = data

    @property
    def tokens(self) -> List[str]:
        return list(self._vocab.keys())

    @property
    def start_id(self) -> int:
        return self.vocab2idx.get(self.start_token)

    @property
    def end_id(self) -> int:
        return self.vocab2idx.get(self.end_token)

    @property
    def vocab2idx(self) -> Dict[str, int]:
        return {word: idx for word, idx in self._vocab.items()}

    @property
    def idx2vocab(self) -> Dict[int, str]:
        return {idx: word for word, idx in self._vocab.items()}

    @property
    def vocab2outcome(self) -> Dict[str, int]:
        return {word: idx for word, idx in self._vocab_outcome.items()}

    @property
    def outcome2vocab(self) -> Dict[int, str]:
        return {idx: word for word, idx in self._vocab_outcome.items()}

    @property
    def vocab_len(self) -> int:
        return len(self.vocab2idx)

    @property
    def num_cases(self):
        return len(self._traces)

    @property
    def ratio_distinct_traces(self):
        return len(set(tuple(tr) for tr in self._traces.values())) / self.num_cases

    @property
    def min_len(self):
        return self._min_len + 2

    @property
    def max_len(self):
        return self._max_len + 2

    @property
    def num_distinct_events(self):
        return len([ev for ev in self.vocab2idx.keys() if ev not in [self.pad_token, self.start_token, self.end_token]])

    def save(self, skip_viz: bool = False, skip_stats: bool = False) -> str:
        target = (self.reader_folder / 'reader.pkl')
        str_target = str(self.reader_folder)
        with target.open('wb') as f:
            pickle.dump(self, f)
            f.close()
        if not skip_stats:
            with (self.reader_folder / 'stats.json').open('w') as f:
                json.dump(self.get_data_statistics(), f, indent=4)
                f.close()
        if not skip_viz:
            self.save_all_viz()
        current = AbstractProcessLogReader.curr_reader_path.open('w')
        current.write(str_target)

        return str_target

    @classmethod
    def load(cls: AbstractProcessLogReader, path: Union[pathlib.Path, str] = None) -> AbstractProcessLogReader:
        if isinstance(path, pathlib.Path):
            path = path
        if type(path) is str:
            path = pathlib.Path(path)
        if path is None:
            path = pathlib.Path(PATH_READERS / cls.__name__)
            if not path.exists():
                latest_reader = cls.curr_reader_path.open('r').read()
                print(f"WARNING: Fallback to latest reader {latest_reader}")
                path = pathlib.Path(latest_reader)

        path = path / 'reader.pkl'
        f = path.open('rb')
        return pickle.load(f)

    # @abstractmethod
    def load_compatible_predictors() -> Tuple[TensorflowModelMixin]:
        pass

    # @abstractmethod
    def load_compatible_generators() -> Tuple[TensorflowModelMixin]:
        pass
    
    def encode(self, df:pd.DataFrame, mode: TaskModes = None, add_start: bool = None, add_end: bool = None)-> np.ndarray:
        data, _ = self.pre_pipeline(df)
        data, _ = self.pipeline.fit_transform(data)
        data, _ = self.post_pipeline(data)
        data = self._move_event_to_end(data, self.col_activity_id)
        data = self._move_outcome_to_end(data, self.col_outcome)
        _traces, _, data = self.group_rows_into_traces(data)
        
        data_container = self._put_data_to_container(_traces)

        initial_data = np.array(data_container)
        features_container, target_container = self._preprocess_containers(TaskModes.OUTCOME_PREDEFINED if not mode else mode, add_start, add_end, initial_data)        
        X, Y = self._generate_dataset_from_ndarray(features_container, target_container, FeatureModes.FULL)
        return X, Y

    def decode_postprocessed(self, df:pd.DataFrame, preprocessors=None, mapping=None, **kwargs):
        df_postprocessed = df.copy()
        mapping = mapping or self.pipeline.mapping
        preprocessors = preprocessors or self.pipeline.collect_as_dict()
        for var_type, columns in mapping.items():
            # for primary_var, sub_var in columns.items():
            #     if sub_var == False:
            #         continue
            if (var_type == CDType.CAT) and (len(columns)):
                ppr = preprocessors.get(CDType.CAT)
                df_postprocessed = ppr.backward(data=df_postprocessed)
            if (var_type == CDType.NUM) and (len(columns)):
                ppr = preprocessors.get(CDType.NUM)
                df_postprocessed = ppr.backward(data=df_postprocessed)
            if (var_type == CDType.BIN) and (len(columns)):
                ppr = preprocessors.get(CDType.BIN)
                df_postprocessed = ppr.backward(data=df_postprocessed)

        df_postprocessed[self.col_activity_id] = df_postprocessed[self.col_activity_id].transform(lambda x: self.idx2vocab[x])        
        return df_postprocessed

    def decode_results(self, events, features, labels, *args):

        cols = {
            9: "case",
            10: "sparcity",
            11: "similarity",
            12: "feasibility",
            13: "delta",
            14: "viability",
        }

        combined = np.concatenate(
            [
                features,
                events[..., None],
                np.repeat(labels[..., None], events.shape[1], axis=1),
                np.repeat(np.arange(len(events))[..., None, None], events.shape[1], axis=1),
                *[np.repeat(arg[..., None], events.shape[1], axis=1) for arg in args],
            ],
            axis=-1,
        )

        df_reconstructed = pd.DataFrame(combined.reshape((-1, combined.shape[-1])))
        df_reconstructed = df_reconstructed.rename(columns={ft.index: ft.col for ft in self.feature_info.all_cols})
        df_reconstructed = df_reconstructed.rename(columns=cols)

        finfo = self.feature_info
        mapping = self.pipeline.mapping
        preprocessors = self.pipeline.collect_as_dict()

        df_postprocessed = df_reconstructed.copy()
        # print(df_postprocessed.shape)
        
        
        for var_type, columns in mapping.items():
            # for primary_var, sub_var in columns.items():
            #     if sub_var == False:
            #         continue
            if (var_type == CDType.CAT) and (len(columns)):
                ppr = preprocessors.get(CDType.CAT)
                df_postprocessed = ppr.backward(data=df_postprocessed)
            if (var_type == CDType.NUM) and (len(columns)):
                ppr = preprocessors.get(CDType.NUM)
                df_postprocessed = ppr.backward(data=df_postprocessed)
            if (var_type == CDType.BIN) and (len(columns)):
                ppr = preprocessors.get(CDType.BIN)
                df_postprocessed = ppr.backward(data=df_postprocessed)

        df_postprocessed[self.col_activity_id] = df_postprocessed[self.col_activity_id].transform(lambda x: self.idx2vocab[x])
        # df_postprocessed[df_postprocessed[self.col_activity_id] == self.pad_token] = None
        # df_postprocessed[df_postprocessed[reader.col_activity_id] == reader.start_token] = None
        # df_postprocessed[df_postprocessed[reader.col_activity_id] == reader.end_token] = None
        return df_postprocessed

    def convert_to_dice4el_format(self, df_post_fa, prefix=""):
        convert_to_dice4el_format = df_post_fa.groupby("id").apply(
            lambda x: {
                prefix + '_' + 'amount': list(x["AMOUNT_REQ"]),
                prefix + '_' + 'activity': list(x[self.col_activity_id]),
                prefix + '_' + 'resource': list(x["Resource"]),
                # prefix + '_' + 'feasibility': list(x.feasibility)[0],
                prefix + '_' + 'label': list(x.label),
                prefix + '_' + 'generator': list(x.generator)[0],
                prefix + '_' + 'id': list(x.id)[0],
            }).to_dict()
        sml = pd.DataFrame(convert_to_dice4el_format).T.reset_index(drop=True)
        return sml

    def zip_fa_with_cf(self, dict_with_cases, rapper_name):
        collector = []
        dict_copy = dict(dict_with_cases.copy())
        factuals = dict_copy.pop("_factuals")
        for idx, (factual, counterfactuals) in enumerate(zip(factuals, dict_copy.get(rapper_name))):
            events, features, llh, viability = factual.all
            df_post_fa = self.decode_results(events, features, llh > 0.5)
            # df_post_fa["feasibility"] = viability.dllh if viability else 0
            df_post_fa["id"] = idx
            df_post_fa["generator"] = rapper_name
            fa_line = self.convert_to_dice4el_format(df_post_fa, "fa")
            for cf_id in range(len(counterfactuals)):
                events, features, llh, viability = counterfactuals[cf_id:cf_id + 1].all
                df_post_cf = self.decode_results(events, features, llh > 0.5)
                # feasibility = viability.dllh
                # feasibility = viability.sparcity
                # feasibility = viability.similarity
                # feasibility = viability.delta
                df_post_cf["id"] = cf_id
                df_post_cf["generator"] = rapper_name
                cf_line = self.convert_to_dice4el_format(df_post_cf, "cf")
                cf_line["feasibility"] = viability.dllh[0][0]
                cf_line["sparcity"] = viability.sparcity[0][0]
                cf_line["similarity"] = viability.similarity[0][0]
                cf_line["delta"] = viability.ollh[0][0]
                cf_line["viability"] = viability.viabs[0][0]

                merged = pd.concat([fa_line, cf_line], axis=1)
                collector.append(merged)

        all_results = pd.concat(collector).sort_values(["feasibility", "viability"], ascending=True)
        return all_results

    def expand_again(self, all_results):
        cols = {
            0: "fa_activity",
            1: "fa_amount",
            2: "fa_resource",
            3: "cf_activity",
            4: "cf_amount",
            5: "cf_resource",
        }
        all_results = pd.DataFrame(all_results.values.T, columns=all_results.keys()) if isinstance(all_results, pd.Series) else all_results
        df_collector = []
        for idx, row in tqdm(all_results.iterrows(), total=len(all_results)):
            tmp_df = pd.DataFrame([
                row["fa_activity"],
                row["fa_amount"],
                row["fa_resource"],
                row["cf_activity"],
                row["cf_amount"],
                row["cf_resource"],
            ]).T
            # tmp_df["fa_amount"] = row["fa_feasibility"]
            tmp_df["generator"] = row["cf_generator"]
            tmp_df["fa_label"] = row["fa_label"]
            tmp_df["cf_label"] = row["cf_label"]
            tmp_df["fa_id"] = row["fa_id"]
            tmp_df["cf_id"] = row["cf_id"]
            tmp_df["cf_feasibility"] = row["feasibility"]
            tmp_df["cf_sparcity"] = row["sparcity"]
            tmp_df["cf_similarity"] = row["similarity"]
            tmp_df["cf_delta"] = row["delta"]
            tmp_df["cf_viability"] = row["viability"]
            df_collector.append(pd.DataFrame(tmp_df))
        new_df = pd.concat(df_collector).rename(columns=cols)
        new_df["cf_resource"] = new_df["cf_resource"]  #.astype(str)
        new_df["fa_resource"] = new_df["fa_resource"]  #.astype(str)
        new_df = new_df.infer_objects()
        return new_df

    def generate_latex_table(self, all_results, index, suffix="", caption=""):
        C_SEQ = "Sequence"
        C_FA = f"Factual {C_SEQ}"
        C_CF = f"Counterfactual {C_SEQ}"
        cols = {
            'fa_activity': (C_FA, "Activity"),
            'fa_amount': (C_FA, "Amount"),
            'fa_resource': (C_FA, "Resource"),
            'fa_label': (C_FA, 'Outcome'),
            'cf_activity': (C_CF, "Activity"),
            'cf_amount': (C_CF, "Amount"),
            'cf_resource': (C_CF, "Resource"),
            'cf_label': (C_CF, 'Outcome'),
        }
        df = all_results.rename(columns=cols).iloc[:, :-7]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.loc[:, [C_FA, C_CF]]
        # something.iloc[:, [1,4]] = something.iloc[:, [1,4]].astype(int)
        # something = something.dropna(axis=0)
        df = df[df.notnull().any(axis=1)]
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace("_", "-").str.replace("None", "")
        df.iloc[:, 4] = df.iloc[:, 4].astype(str).str.replace("_", "-").str.replace("None", "")
        df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace(".0", "").str.replace("None", "")
        df.iloc[:, 6] = df.iloc[:, 6].astype(str).str.replace(".0", "").str.replace("None", "")
        # df = df[~(df[(C_FA, "Resource")]=="nan")]

        df_styled = df.style.format(
            # escape='latex',
            precision=0,
            na_rep='',
            thousands=" ",
        ).hide(None)

        df_latex = df_styled.to_latex(
            multicol_align='l',
            # column_format='l',
            caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
            label=f"tbl:example-cf-{suffix}",
            hrules=True,
        )
        return df, df_styled, df_latex


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
        self.important_cols = self.important_cols.set_col_timestamp("time:timestamp").set_col_activity_id("concept:name").set_col_case_id("case:concept:name")

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
        return super().init_meta()


def test_dataset(reader: AbstractProcessLogReader, batch_size=42, ds_mode: DatasetModes = None, tg_mode: TaskModes = None, ft_mode: FeatureModes = None):
    def show_instance(data_point):

        if type(data_point[0]) == tuple:
            print("FEATURES")
            print(data_point[0][0].shape)
            print(data_point[0][1].shape)
        else:
            print("FEATURES")
            print(data_point[0].shape)

        print("TARGET")
        print(data_point[1].shape)
        print(f'----------------------------------------------------------------')

    for tg in TaskModes if tg_mode is None else [tg_mode]:
        print(f"================= {tg.name} =======================")
        if tg in [TaskModes.ENCODER_DECODER, TaskModes.OUTCOME_PREDEFINED]:
            print(f"Skip {tg}")
            continue
        params = it.product(DatasetModes if ds_mode is None else [ds_mode], FeatureModes if ft_mode is None else [ft_mode])
        for ds, ft in params:
            print(f"-------------------------------- {ds.name} - {ft.name} --------------------------------")
            reader = reader.instantiate_dataset(tg)
            data = reader.get_dataset(batch_size, ds, ft).take(2)
            data_point = next(iter(data))
            show_instance(data_point)


if __name__ == '__main__':
    reader = AbstractProcessLogReader(
        log_path=DATA_FOLDER / 'dataset_bpic2020_tu_travel/RequestForPayment.xes',
        csv_path=DATA_FOLDER_PREPROCESSED / 'RequestForPayment.csv',
        mode=TaskModes.OUTCOME_EXTENSIVE_DEPRECATED,
    )
    # data = data.init_log(save=0)
    reader = reader.init_meta()
    test_dataset(reader, 42, ds_mode=DatasetModes.TRAIN, tg_mode=None, ft_mode=FeatureModes.FULL)
    print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    features, targets = reader._prepare_input_data(reader.trace_test[0:2], reader.target_test[0:2])
    print(reader.decode_matrix(features[0]))
    print(reader.get_data_statistics())