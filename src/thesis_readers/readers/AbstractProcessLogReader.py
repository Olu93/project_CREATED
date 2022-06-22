from __future__ import annotations
from abc import ABC, abstractmethod
from collections import UserList

import io
import itertools as it
import json
import os
import pathlib
from enum import IntEnum
from typing import Counter, Dict, Iterable, Iterator, List, Sequence, Union

from thesis_commons.distributions import DataDistribution
from thesis_readers.helper.preprocessing import BinaryEncodeOperation, CategoryEncodeOperation, DropOperation, IrreversableOperation, LabelEncodeOperation, NumericalEncodeOperation, Operation, ProcessingPipeline, ReversableOperation, SetIndexOperation, StandardOperations, TimeExtractOperation
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
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.lm import \
    preprocessing as \
    nltk_preprocessing  # https://www.kaggle.com/alvations/n-gram-language-model-with-nltk
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

from thesis_commons import random
from thesis_commons.constants import PATH_READERS
from thesis_commons.decorators import collect_time_stat
from thesis_commons.functions import (reverse_sequence_2, shift_seq_backward, shift_seq_forward)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER, DATA_FOLDER_PREPROCESSED, DATA_FOLDER_VISUALIZATION)

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG
# TODO: Maaaaaybe... Put into thesis_commons  -- NO IT FITS RIGHT HERE
# TODO: Checkout TimeseriesGenerator https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
# TODO: Checkout Sampling Methods    https://medium.com/deep-learning-with-keras/sampling-in-text-generation-b2f4825e1dad
# TODO: Add ColStats class with appropriate encoders

np.set_printoptions(edgeitems=26, linewidth=1000)


class AbstractProcessLogReader():
    """DatasetBuilder for my_dataset dataset."""

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
                 debug=False,
                 mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE,
                 max_tokens: int = None,
                 ngram_order: int = 2,
                 **kwargs) -> None:
        super(AbstractProcessLogReader, self).__init__(**kwargs)
        self.log = None
        self.log_path: str = None
        self.data: pd.DataFrame = None
        self.debug: bool = False
        self.col_case_id: str = None
        self.col_activity_id: str = None
        self._vocab: dict = None
        self.mode: TaskModes = None
        self._original_data: pd.DataFrame = None
        self.debug = debug
        self.mode = mode
        self.log_path = pathlib.Path(log_path)
        self.csv_path = pathlib.Path(csv_path)
        self.col_case_id = col_case_id
        self.col_activity_id = col_event_id
        self.col_timestamp = col_timestamp
        self.col_outcome: str = None
        self.preprocessors = {}
        self.ngram_order = ngram_order
        self.reader_folder: pathlib.Path = (PATH_READERS / type(self).__name__).absolute()
        self.pipeline = ProcessingPipeline()
        self.data_distribution: DataDistribution = None

        if not self.reader_folder.exists():
            os.mkdir(self.reader_folder)

    @collect_time_stat
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

    @collect_time_stat
    def init_meta(self, skip_dynamics: bool = False):
        is_from_log = self._original_data is not None
        self.col_case_id = self.col_case_id if is_from_log else 'case:concept:name'
        self.col_activity_id = self.col_activity_id if is_from_log else 'concept:name'
        self.col_timestamp = self.col_timestamp if is_from_log else 'time:timestamp'
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
        self.col_stats = self._initialize_col_stats(self._original_data)
        self.data, self.pipeline = self.preprocess()
        self.data = self._move_outcome_to_end(self.data, self.col_outcome)
        self.register_vocabulary()
        self.group_rows_into_traces()
        self.gather_information_about_traces()
        if not skip_dynamics:
            self.compute_trace_dynamics()
        if self.mode is not None:
            self.instantiate_dataset(self.mode)

        return self

    def _move_outcome_to_end(self, data: pd.DataFrame, col_outcome):
        cols = list(data.columns.values)
        cols.pop(cols.index(col_outcome))
        return data[cols + [self.col_outcome]]

    # @staticmethod
    # def gather_grp_column_statsitics(df: pd.DataFrame):
    #     full_len = len(df)
    #     return {col: {'name': col, 'entropy': entropy(df[col].value_counts()), 'dtype': df[col].dtype, 'missing_ratio': df[col].isna().sum() / full_len} for col in df.columns}

    def _initialize_col_stats(self, data: pd.DataFrame, na_val='missing', min_diversity=0.0, max_diversity=0.8, max_similarity=0.6, max_missing=0.75):

        data = data.replace(na_val, np.nan) if na_val else data
        skip = [self.col_case_id, self.col_activity_id, self.col_timestamp, self.col_outcome]
        col_statistics = self._gather_column_statsitics(data)
        col_statistics = {
            col: {
                **stats, "uselessness": self._is_useless_col(stats, min_diversity, max_diversity, max_similarity, max_missing)
            }
            for col, stats in col_statistics.items()
        }

        col_statistics = {col: {**stats, "is_useless_cadidate": any(stats["uselessness"].values()) and (not stats.get("is_timestamp"))} for col, stats in col_statistics.items()}
        col_statistics = {col: {**stats, "is_useless": stats["is_useless_cadidate"] if (col not in skip) else False} for col, stats in col_statistics.items()}

        return col_statistics

    def phase_1_premature_drop(self, data: pd.DataFrame, cols=None):

        if not cols:
            return data, list(data.columns)
        new_data = data.drop(cols, axis=1)
        # removed_cols = set(data.columns) - set(new_data.columns)
        return new_data, list(new_data.columns)

    def phase_2_stat_drop(self, data: pd.DataFrame, col_statistics=None):

        if not col_statistics:
            return data, list(data.columns)
        cols = [col for col, val in col_statistics.items() if val["is_dropped"]]
        new_data = data.drop(cols, axis=1)
        # removed_cols = set(data.columns) - set(new_data.columns)
        return new_data, list(new_data.columns)

    def phase_3_time_extract(self, data: pd.DataFrame, col_timestamp=None):
        time_vals = data[col_timestamp].dt.isocalendar().week
        if time_vals.nunique() > 1:
            data["timestamp.week"] = time_vals

        time_vals = data[col_timestamp].dt.weekday
        if time_vals.nunique() > 1:
            data["timestamp.weekday"] = time_vals

        time_vals = data[col_timestamp].dt.day
        if time_vals.nunique() > 1:
            data["timestamp.day"] = time_vals

        time_vals = data[col_timestamp].dt.hour
        if time_vals.nunique() > 1:
            data["timestamp.hour"] = time_vals

        time_vals = data[col_timestamp].dt.minute
        if time_vals.nunique() > 1:
            data["timestamp.minute"] = time_vals

        time_vals = data[col_timestamp].dt.second
        if time_vals.nunique() > 1:
            data["timestamp.second"] = time_vals

        all_time_cols = data.filter(regex='timestamp\..+').columns.tolist()
        return data.drop(col_timestamp, axis=1), tuple(all_time_cols)

    def phase_4_set_index(self, data: pd.DataFrame, col_case_id=None):
        if col_case_id is None:
            return data
        return data.set_index(col_case_id)

    def phase_4_label_encoding(self, data: pd.DataFrame, cols=[]):
        preprocessors = {}
        for col in cols:
            preprocessors[col] = preprocessing.LabelEncoder().fit(data[col])
            data[col] = preprocessors[col].transform(data[col])
        return data, preprocessors

    def phase_5_numeric_standardisation(self, data: pd.DataFrame, cols=[]):
        preprocessors = {}
        if not len(cols):
            return data, preprocessors
        encoder = preprocessing.StandardScaler()
        preprocessors['numericals'] = encoder
        data[list(cols)] = encoder.fit_transform(data[list(cols)])
        return data, preprocessors

    def phase_5_binary_encode(self, data: pd.DataFrame, cols=[]):
        preprocessors = {}
        if not len(cols):
            return data, preprocessors
        encoder = preprocessing.OneHotEncoder(drop='if_binary', sparse=False)
        preprocessors['binaricals'] = encoder
        cols_all = list(cols)
        new_data = encoder.fit_transform(data[cols_all])
        data = data.drop(cols_all, axis=1)
        data[cols_all] = new_data
        return data, preprocessors

    def phase_5_cat_encode(self, data: pd.DataFrame, cols=[]):
        preprocessors = {}
        if not len(cols):
            return data, preprocessors
        encoder = ce.BaseNEncoder(return_df=True, drop_invariant=True, base=2)
        preprocessors['categoricals'] = encoder
        cols_all = list(cols)
        new_data = encoder.fit_transform(data[cols_all].astype(str))
        data = data.drop(cols_all, axis=1)
        data[new_data.columns] = new_data
        return data, preprocessors

    def phase_6_normalisation(self, data: pd.DataFrame, cols=[]):
        preprocessors = {}
        if not len(cols):
            return data, preprocessors
        encoder = preprocessing.StandardScaler()
        preprocessors['all'] = encoder
        cols_all = list(cols)
        new_data = encoder.fit_transform(data[cols_all])
        data = data.drop(cols_all, axis=1)
        data[cols_all] = new_data
        return data, preprocessors

    def phase_end_postprocess(self, data: pd.DataFrame, **kwargs):
        return data

    @collect_time_stat
    def preprocess_data(self, data: pd.DataFrame, col_stats: Dict, **kwargs):
        remove_cols = kwargs.get('remove_cols', [])
        dropped_by_stats_cols = [col for col, val in col_stats.items() if (val["is_useless"]) and (col not in remove_cols)]
        col_binary_all = list([col for col, stats in col_stats.items() if stats.get("is_binary") and not stats.get("is_outcome")])
        col_cat_all = list([col for col, stats in col_stats.items() if stats.get("is_categorical") and not (stats.get("is_col_case_id") or stats.get("is_col_activity_id"))])
        col_numeric_all = list([col for col, stats in col_stats.items() if stats.get("is_numeric")])
        col_timestamp_all = list([col for col, stats in col_stats.items() if stats.get("is_timestamp")])

        op1 = DropOperation(remove_cols, name="premature_drop")
        op2 = op1.chain(DropOperation(dropped_by_stats_cols, name="usability_drop"))
        op3 = op2.chain(SetIndexOperation([self.col_case_id], name="set_index"))
        op4 = op3.chain(TimeExtractOperation(col_timestamp_all, name="temporals"))
        op4 = op4.append_next(BinaryEncodeOperation(col_binary_all, name="binaricals")).append_next(CategoryEncodeOperation(col_cat_all, name="categoricals")).append_next(
            NumericalEncodeOperation(col_numeric_all + col_timestamp_all, name="numericals"))

        pipeline = ProcessingPipeline().set_root(op1).fit(data, **kwargs)
        # distribution_mappings = {
        #     "temporals": self.pipeline["time_encoding"].pre2post,
        #     "categoricals": self.pipeline["category_encoding"].pre2post,
        #     "binaricals": self.pipeline["binary_encoding"].pre2post,
        #     "numericals": self.pipeline["numeric_encoding"].pre2post,
        # }
        return data, pipeline

    @collect_time_stat
    def preprocess_level_general(self, remove_cols=None, max_diversity_thresh=0.75, min_diversity=0.0, too_similar_thresh=0.6, missing_thresh=0.75, **kwargs):
        # self.data = self.original_data
        # if remove_cols:
        #     self.data = self.data.drop(remove_cols, axis=1)
        # col_statistics = self._gather_column_statsitics(self.data.select_dtypes('object'))
        # col_statistics = {
        #     col: dict(stats, is_useless=self._is_useless_col(stats, min_diversity, max_diversity_thresh, too_similar_thresh, missing_thresh))
        #     for col, stats in col_statistics.items()
        #     if col not in [self.col_case_id, self.col_activity_id, self.col_timestamp] + ([self.col_outcome] if hasattr(self, "col_outcome") else [])
        # }
        # col_statistics = {col: dict(stats, is_dropped=any(stats["is_useless"])) for col, stats in col_statistics.items()}
        # cols_to_remove = [col for col, val in col_statistics.items() if val["is_dropped"]]
        # self.data = self.data.drop(cols_to_remove, axis=1)
        pass

    def _is_useless_col(self, stats, min_diversity, max_diversity, max_similarity, max_missing):
        is_singular = stats.get("diversity") == 0
        is_diverse = stats.get("diversity") > min_diversity
        is_not_diverse_enough = bool((not is_diverse) & (not stats.get('is_numeric')))
        is_unique_to_case = bool(stats.get("intracase_similarity") > max_similarity)
        is_missing_too_many = bool(stats.get("missing_ratio") > max_missing)
        return {
            "is_singular": is_singular,
            "is_not_diverse_enough": is_not_diverse_enough,
            "is_unique_to_case": is_unique_to_case,
            "is_missing_too_many": is_missing_too_many,
        }

    def _gather_column_statsitics(self, df: pd.DataFrame):
        full_len = len(df)
        num_traces = df[self.col_case_id].nunique(False)
        results = {
            col: {
                'name': col,
                'diversity': df[col].nunique(False) / full_len if df[col].nunique(False) > 1 else 0,  # Special case of just one unique
                'dtype': str(df[col].dtype),
                'missing_ratio': df[col].isna().sum() / full_len,
                'intracase_similarity': 1 - (np.abs(df[col].nunique(False) - num_traces) / np.max([df[col].nunique(False), num_traces])),
                '_num_unique': df[col].nunique(False),
                'is_numeric': bool(self._is_numeric(df[col])),
                'is_binary': bool(self._is_binary(df[col])),
                'is_categorical': bool(self._is_categorical(df[col])),
                'is_timestamp': bool(self._is_timestamp(df[col])),
                'is_singular': bool(self._is_singular(df[col])),
                'is_col_case_id': self.col_case_id == col,
                'is_col_timestamp': self.col_timestamp == col,
                'is_col_outcome': self.col_outcome == col,
                'is_col_activity_id': self.col_activity_id == col,
                'is_case_id': self.col_case_id == col,
                '_num_rows': full_len,
                '_num_traces': num_traces,
            }
            for col in df.columns
        }
        return results

    def _is_categorical(self, series):
        return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and series.nunique(False) > 2

    def _is_binary(self, series):
        return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and series.nunique(False) == 2

    def _is_singular(self, series):
        return series.nunique(False) == 1

    def _is_numeric(self, series):
        return pd.api.types.is_numeric_dtype(series)

    def _is_timestamp(self, series):
        if (series.name == "second_tm") or (series.name == self.col_timestamp):
            print("STOP")
        return pd.api.types.is_datetime64_any_dtype(series)

    @collect_time_stat
    def preprocess(self, **kwargs):
        return self.preprocess_data(self._original_data, self.col_stats)

    @collect_time_stat
    def register_vocabulary(self):
        all_unique_tokens = list(self.data[self.col_activity_id].unique())

        self._vocab = {word: idx for idx, word in enumerate(all_unique_tokens, 1)}
        self._vocab[self.pad_token] = 0
        self._vocab[self.start_token] = len(self._vocab)
        self._vocab[self.end_token] = len(self._vocab)

    @collect_time_stat
    def group_rows_into_traces(self):
        self.data = self.data.replace({self.col_activity_id: self._vocab})
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
        self.idx_event_attribute = self.data.columns.get_loc(self.col_activity_id)
        self.idx_outcome = self.data.columns.get_loc(self.col_outcome) if self.col_outcome is not None else None
        self._idx_time_attributes = {col: self.data.columns.get_loc(col) for col in self.col_timestamp_all}
        skip = [self.col_activity_id, self.col_case_id, self.col_timestamp, self.col_outcome]
        self._idx_features = {col: self.data.columns.get_loc(col) for col in self.data.columns if col not in skip}
        self._idx_distribution = {
            vartype: {var: [self.data.columns.get_loc(col) for col in grp]
                      for var, grp in grps.items()}
            for vartype, grps in self.distribution_mappings.items()
        }
        self.num_event_attributes = len(self._idx_features)
        # self.feature_shapes = ((self.max_len, ), (self.max_len, self.feature_len - 1), (self.max_len, self.feature_len), (self.max_len, self.feature_len))
        # self.feature_types = (tf.float32, tf.float32, tf.float32, tf.float32)

        # self.distinct_trace_counts[(self.start_id,)] = self.log_len
        # self.distinct_trace_weights = {tr: 1 / val for tr, val in self.distinct_trace_counts.items()}
        # self.distinct_trace_weights = {tr: sum(list(self.distinct_trace_count.values())) / val for tr, val in self.distinct_trace_count.items()}

    def get_data_statistics(self):
        return {
            "class_name": type(self).__name__,
            "num_cases": self.num_cases,
            "min_seq_len": self._min_len,
            "max_seq_len": self._max_len,
            "ratio_distinct_traces": self.ratio_distinct_traces,
            "num_distinct_events": self.num_distinct_events,
            "num_data_columns": self.num_data_cols,
            "num_event_features": self.num_event_attributes,
            "length_distribution": self.length_distribution,
            "time": dict(time_unit="seconds", **self.time_stats),
            "column_stats": self.col_stats,
            "orig": {
                "max_seq_len": self._orig_max_len,
                "length_distribution": self._orig_length_distribution,
            }
        }

    @collect_time_stat
    def compute_trace_dynamics(self):

        print("START computing process dynamics")
        self._traces_only_events = {idx: df[self.col_activity_id].values.tolist() for idx, df in self.grouped_traces}
        self.features_by_actvity = {activity: df.drop(self.col_activity_id, axis=1) for activity, df in list(self.data.groupby(by=self.col_activity_id))}

        self._traces_only_events_txt = {idx: [self.idx2vocab[i] for i in indices] for idx, indices in self._traces_only_events.items()}
        self.trace_counts = Counter(tuple(trace[:idx + 1]) for trace in self._traces_only_events.values() for idx in range(len(trace)))
        self.trace_counts_by_length = {length: Counter({trace: count for trace, count in self.trace_counts.items() if len(trace) == length}) for length in range(self.max_len)}
        self.trace_counts_by_length_sums = {length: sum(counter.values()) for length, counter in self.trace_counts_by_length.items()}
        self.trace_probs_by_length = {
            length: {trace: count / self.trace_counts_by_length_sums.get(length, 0)
                     for trace, count in counter.items()}
            for length, counter in self.trace_counts_by_length.items()
        }
        self.trace_counts_sum = sum(self.trace_counts.values())
        self.trace_probs = {trace: counts / self.trace_counts_sum for trace, counts in self.trace_counts.items()}
        self.trace_ngrams_hard = MLE(self.ngram_order)
        self.trace_ngrams_soft = KneserNeyInterpolated(self.ngram_order)
        self.trace_ngrams_hard.fit(*nltk_preprocessing.padded_everygram_pipeline(self.ngram_order, list(self._traces_only_events_txt.values())))
        self.trace_ngrams_soft.fit(*nltk_preprocessing.padded_everygram_pipeline(self.ngram_order, list(self._traces_only_events_txt.values())))
        print("END computing process dynamics")

    @collect_time_stat
    def instantiate_dataset(self, mode: TaskModes = None, add_start: bool = None, add_end: bool = None):
        # TODO: Add option to mirror train and target
        # TODO: Add option to add boundary tags
        print("Preprocess data")
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

            mask = np.not_equal(features_container[:, :, self.idx_event_attribute], 0)
            target_container = all_next_activities[:, -1][:, None]
            extensive_out_come = mask * target_container
            events_per_row = np.count_nonzero(features_container[:, :, self.idx_event_attribute], axis=-1)
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
            events = features_container[:, :, self.idx_event_attribute]
            events = np.roll(events, 1, axis=1)
            events[:, -1] = self.end_id

            all_rows = [list(row[np.nonzero(row)]) for row in events]
            all_splits = [(idx, split) for idx, row in enumerate(all_rows) if len(row) > 1 for split in [random.integers(1, len(row) - 1)]]

            features_container = [all_rows[idx][:split] for idx, split in all_splits]
            target_container = [all_rows[idx][split:] for idx, split in all_splits]

        if self.mode == TaskModes.OUTCOME_PREDEFINED:
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
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

            mask = np.not_equal(features_container[:, :, self.idx_event_attribute], 0)
            target_container = all_next_activities[:, -1][:, None]
            extensive_out_come = mask * target_container

        return features_container, target_container

    def _put_data_to_container(self):
        data_container = np.zeros([self.log_len, self.max_len, self.num_data_cols])
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
            results[:, -1, self.idx_event_attribute] = self.end_id
            results = reverse_sequence_2(results)
            results = np.roll(results, 1, axis=1)
            results[:, 0, self.idx_event_attribute] = self.start_id
            results = reverse_sequence_2(results)
            results[:, -1, self.idx_event_attribute] = 0
            results = np.roll(results, 1, axis=1)
            del self._vocab[self.end_token]
            return results
        if ((not start_tag) and end_tag):
            results[:, -1, self.idx_event_attribute] = self.end_id
            del self._vocab[self.start_token]
            self._vocab[self.end_token] = len(self._vocab)
            return results
        if (start_tag and end_tag):
            results[:, -1, self.idx_event_attribute] = self.end_id
            results = reverse_sequence_2(results)
            results = np.roll(results, 1, axis=1)
            results[:, 0, self.idx_event_attribute] = self.start_id
            results = reverse_sequence_2(results)
            return results

    def _reverse_sequence(self, data_container):
        original_data = np.array(data_container)
        flipped_data = np.flip(data_container, axis=1)
        results = np.zeros_like(original_data)
        results[np.nonzero(original_data.sum(-1) != 0)] = flipped_data[(flipped_data.sum(-1) != 0) == True]
        return results

    def _get_events_only(self, data_container, shift=None):
        result = np.array(data_container[:, :, self.idx_event_attribute])
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
            res_features = (features[:, :, self.idx_event_attribute], np.zeros_like(features[:, :, self.idx_features]))
        if ft_mode == FeatureModes.FEATURE:
            res_features = (np.zeros_like(features[:, :, self.idx_event_attribute]), features[:, :, self.idx_features])
        if ft_mode == FeatureModes.TIME:
            res_features = (features[:, :, self.idx_event_attribute], features[:, :, self.idx_time_attributes])
        if ft_mode == FeatureModes.FULL:
            res_features = (features[:, :, self.idx_event_attribute], features[:, :, self.idx_features])

        if not ft_mode == FeatureModes.ENCODER_DECODER:
            self.num_event_attributes = res_features.shape[-1] if not type(res_features) == tuple else res_features[1].shape[-1]
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

    def get_dataset_generative(self, ds_mode: DatasetModes, ft_mode: FeatureModes, batch_size=1, num_data: int = None, flipped_target=False):
        # TODO: Maybe return Population object instead also rename population to Cases
        res_data, res_targets = self._generate_dataset(ds_mode, ft_mode)
        flipped_res_features = (reverse_sequence_2(res_data[0]), reverse_sequence_2(res_data[1]))

        results = (res_data, flipped_res_features if flipped_target else res_data)

        self.num_event_attributes = res_data[1].shape[-1]
        dataset = tf.data.Dataset.from_tensor_slices(results).batch(batch_size)
        dataset = dataset.take(num_data) if num_data else dataset

        return dataset

    def get_dataset_example(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TRAIN, ft_mode: FeatureModes = FeatureModes.FULL):
        pass

    def get_dataset_with_indices(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TEST, ft_mode: FeatureModes = FeatureModes.FULL):
        collector = []
        dataset = None
        # dataset = self.get_dataset(1, data_mode, ft_mode)
        trace, target = self._choose_dataset_shard(data_mode)
        res_features, res_targets = self._prepare_input_data(trace, target, ft_mode)
        res_features, res_targets, res_sample_weights = self._attach_weight((res_features, res_targets), res_targets)
        res_indices = trace[:, :, self.idx_event_attribute].astype(np.int32)
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
    def idx_time_attributes(self):
        return list(self._idx_time_attributes.values())

    @property
    def idx_features(self):
        return list(self._idx_features.values())

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
    def start_id(self) -> str:
        return self.vocab2idx.get(self.start_token)

    @property
    def end_id(self) -> str:
        return self.vocab2idx.get(self.end_token)

    @property
    def pad_id(self) -> str:
        return self.vocab2idx[self.pad_token]

    @property
    def vocab2idx(self) -> Dict[str, int]:
        return {word: idx for word, idx in self._vocab.items()}

    @property
    def idx2vocab(self) -> Dict[int, str]:
        return {idx: word for word, idx in self._vocab.items()}

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
        if type(path) is pathlib.Path:
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