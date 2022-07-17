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
    np.set_printoptions(edgeitems=26, linewidth=1000, precision=8)
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
        self.group_rows_into_traces()
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
        self._vocab_outcome = {word: idx for idx, word in enumerate(all_unique_outcomes)}

    @collect_time_stat
    def group_rows_into_traces(self):
        self.data = self.data.replace({self.col_activity_id: self._vocab})
        self.data = self.data.replace({self.col_outcome: self._vocab_outcome})
        self.grouped_traces = list(self.data.groupby(by=self.col_case_id))
        self._traces = {idx: df for idx, df in self.grouped_traces}

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
        self.outcome_distribution = Counter(self.data.groupby(self.col_case_id).nth(0)[self.feature_info.col_outcome].replace({
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
        self.data_container = self._put_data_to_container()
        # self.data_container[idx, -1, self.idx_event_attribute] = self.vocab2idx[self.end_token]
        # self.data_container[idx, -df_end - 1, self.idx_event_attribute] = self.vocab2idx[self.start_token]

        initial_data = np.array(self.data_container)
        features_container, target_container = self._preprocess_containers(self.mode, add_start, add_end, initial_data)
        self.traces_preprocessed = features_container, target_container
        self.traces, self.targets = self.traces_preprocessed
        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
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

    def _put_data_to_container(self):
        # data_container = np.empty([self.log_len, self.max_len, self.num_data_cols])
        # data_container = np.ones([self.log_len, self.max_len, self.num_data_cols]) * -42
        data_container = np.ones([self.log_len, self.max_len, self.num_data_cols]) * self.pad_value
        # data_container[:, :, self.idx_event_attribute] = self.pad_id
        # data_container[:, :, self.idx_features] = self.pad_value
        # data_container[:] = np.nan
        loader = tqdm(self._traces.items(), total=len(self._traces))
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

    def get_dataset_generative(self, ds_mode: DatasetModes, ft_mode: FeatureModes, batch_size=1, num_data: int = None, flipped_input=False, flipped_output=False):
        # TODO: Maybe return Population object instead also rename population to Cases
        data_input, data_target = self._generate_dataset(ds_mode, ft_mode)

        results = (
            (reverse_sequence_2(data_input[0]), reverse_sequence_2(data_input[1])) if flipped_input else data_input,
            (reverse_sequence_2(data_input[0]), reverse_sequence_2(data_input[1])) if flipped_output else data_input,
        )

        self.feature_len = data_input[1].shape[-1]
        dataset = tf.data.Dataset.from_tensor_slices(results).batch(batch_size)
        dataset = dataset.take(num_data) if num_data else dataset

        return dataset

    # def get_distribution(self, ds_mode: DatasetModes, ft_mode: FeatureModes):
    #     res_data, res_targets = self._generate_dataset(ds_mode, ft_mode)
    #     DataDistribution

    def get_dataset_example(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TRAIN, ft_mode: FeatureModes = FeatureModes.FULL):
        pass

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

    def gather_full_dataset(self, dataset: tf.data.Dataset):
        collector = []
        for data_point in dataset:
            instance = []
            for part in data_point:
                if type(part) in [tuple, list] and len(part) == 2 and len(part[0]) > 2:
                    instance.extend(part)
                else:
                    instance.append(part)
            collector.append(instance)
        stacked_all_stuff = [np.stack(tmp) for tmp in zip(*collector)]
        # Until -2 to ignore sample weight
        return stacked_all_stuff[0], stacked_all_stuff[1:-2], stacked_all_stuff[-2], stacked_all_stuff[-1]

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