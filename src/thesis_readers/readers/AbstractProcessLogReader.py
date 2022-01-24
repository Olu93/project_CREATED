import time
import random
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
from thesis_readers.helper.modes import DatasetModes, FeatureModes, TaskModes
from imblearn.over_sampling import RandomOverSampler

from thesis_readers.helper.constants import DATA_FOLDER, DATA_FOLDER_PREPROCESSED, DATA_FOLDER_VISUALIZATION

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG


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
    mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE
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
                 mode: TaskModes = TaskModes.NEXT_EVENT_EXTENSIVE,
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
        self.data["timestamp.month"] = self.data[self.col_timestamp].dt.month
        self.data["timestamp.week"] = self.data[self.col_timestamp].dt.isocalendar().week
        self.data["timestamp.weekday"] = self.data[self.col_timestamp].dt.weekday
        self.data["timestamp.day"] = self.data[self.col_timestamp].dt.day
        self.data["timestamp.hour"] = self.data[self.col_timestamp].dt.hour
        self.data["timestamp.minute"] = self.data[self.col_timestamp].dt.minute
        self.data["timestamp.second"] = self.data[self.col_timestamp].dt.second
        del self.data[self.col_timestamp]
        num_encoder = StandardScaler()
        self.col_timestamp_all = self.data.filter(regex='timestamp.+').columns.tolist()
        self.preprocessors['time'] = num_encoder
        self.data[self.col_timestamp_all] = num_encoder.fit_transform(self.data[self.col_timestamp_all])
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
        self.idx_time_attributes = [self.data.columns.get_loc(col) for col in self.col_timestamp_all]
        self.idx_features = [self.data.columns.get_loc(col) for col in self.data.columns if col not in [self.col_activity_id, self.col_case_id, self.col_timestamp]]
        # self.feature_shapes = ((self.max_len, ), (self.max_len, self.feature_len - 1), (self.max_len, self.feature_len), (self.max_len, self.feature_len))
        # self.feature_types = (tf.float32, tf.float32, tf.float32, tf.float32)
        # self._traces_only_events = {idx: df[self.col_activity_id].values.tolist() for idx, df in self.grouped_traces}
        # self.distinct_trace_counts = Counter(tuple(trace[:idx+1]) for trace in self._traces_only_events.values() for idx in range(len(trace)))
        # self.distinct_trace_counts[(self.start_id,)] = self.log_len
        # self.distinct_trace_weights = {tr: 1 / val for tr, val in self.distinct_trace_counts.items()}
        # self.distinct_trace_weights = {tr: sum(list(self.distinct_trace_count.values())) / val for tr, val in self.distinct_trace_count.items()}

    def instantiate_dataset(self):
        print("Preprocess data")
        self.data_container = np.zeros([self.log_len, self.max_len, self.feature_len])
        loader = tqdm(self._traces.items(), total=len(self._traces))
        group_indices = None
        for idx, (case_id, df) in enumerate(loader):
            df_end = len(df)
            # if df_end <= 1:
            #     print(idx)
            self.data_container[idx, -df_end:] = df.values
            # self.data_container[idx, -1, self.idx_event_attribute] = self.vocab2idx[self.end_token]
            self.data_container[idx, -df_end - 1, self.idx_event_attribute] = self.vocab2idx[self.start_token]

        if self.mode == TaskModes.NEXT_EVENT_EXTENSIVE:
            all_next_activities = self._get_next_activities()
            self.traces = self.data_container, all_next_activities

        if self.mode == TaskModes.NEXT_EVENT:
            all_next_activities = self._get_next_activities()  # 9 is missing
            tmp = [(ft[:idx + 1], tg[idx]) for ft, tg in zip(self.data_container, all_next_activities) for idx in range(len(ft)) if (ft[:idx + 1].sum() != 0) and (tg[idx] != 0)]
            # tmp2 = list(zip(*tmp))
            features_container = np.zeros([len(tmp), self.max_len, self.feature_len])
            target_container = np.zeros([len(tmp), 1], dtype=np.int32)
            for idx, (ft, tg) in enumerate(tmp):
                features_container[idx, -len(ft):] = ft
                target_container[idx] = tg
            self.traces = features_container, target_container

        if self.mode == TaskModes.ENCODER_DECODER:
            # TODO: Include extensive version of enc dec (maybe if possible)
            events = self.data_container[:, :, self.idx_event_attribute]
            events = np.roll(events, 1, axis=1)
            events[:, -1] = self.end_id

            all_rows = [list(row[np.nonzero(row)]) for row in events]
            all_splits = [(idx, split) for idx, row in enumerate(all_rows) if len(row) > 1 for split in [random.randint(1, len(row) - 1)]]

            features_container = [all_rows[idx][:split] for idx, split in all_splits]
            target_container = [all_rows[idx][split:] for idx, split in all_splits]
            self.traces = features_container, target_container

        if self.mode == TaskModes.ENCODER_DECODER_PADDED:
            # TODO: Include extensive version of enc dec (maybe if possible)
            self.data_container = np.roll(self.data_container, -1, axis=1)
            self.data_container[:, -1, self.idx_event_attribute] = self.end_id
            events = self.data_container[:, :, self.idx_event_attribute]
            starts = np.not_equal(events, 0).argmax(-1)
            ends = (np.ones_like(starts) * self.max_len)
            lenghts = ends - starts
            all_splits = [(
                idx,
                start,
                split,
                end,
            ) for idx, start, end, le in zip(range(len(starts)), starts, ends, lenghts) if le > 1 for split in [random.randint(1, le - 1)]]

            features_container = np.zeros([len(all_splits), self.max_len, self.feature_len])
            target_container = np.zeros((len(all_splits), self.max_len), dtype=np.int32)
            for row_num, (idx, start, gap, end) in enumerate(all_splits):
                features_container[row_num, -gap:end] = self.data_container[idx, start:start + gap]
                target_container[row_num, 0:end - (start + gap)] = events[idx, start + gap:end]
            self.traces = features_container, target_container

        # if self.mode == TaskModes.EXTENSIVE:
        #     self.traces = ([tr[0:end - 1], tr[1:end]] for tr in loader for end in range(2, len(tr) + 1) if len(tr) > 1)

        # if self.mode == TaskModes.EXTENSIVE_RANDOM:
        #     tmp_traces = [tr[random.randint(0, len(tr) - 1):] for tr in loader for sample in self._heuristic_bounded_sample_size(tr) if len(tr) > 1]
        #     self.traces = [tr[:random.randint(2, len(tr))] for tr in tqdm(tmp_traces, desc="random-samples") if len(tr) > 1]

        if self.mode == TaskModes.OUTCOME:
            all_next_activities = self._get_next_activities()
            self.data_container = np.roll(self.data_container, 1, axis=1)
            self.data_container[:, 0] = 0
            end_positions = (all_next_activities == self.end_id).argmax(-1)[:, None]
            out_come = np.take_along_axis(all_next_activities, end_positions - 1, axis=1)  # ATTENTION .reshape(-1)
            self.traces = self.data_container, out_come

        if self.mode == TaskModes.OUTCOME_EXTENSIVE:
            # TODO: Design features like next event
            all_next_activities = self._get_next_activities()
            self.data_container = np.roll(self.data_container, 1, axis=1)
            self.data_container[:, 0] = 0
            mask = np.not_equal(self.data_container[:, :, self.idx_event_attribute], 0)
            out_come = all_next_activities[:, -2][:, None]
            extensive_out_come = mask * out_come
            self.traces = self.data_container, extensive_out_come

        self.traces, self.targets = self.traces
        # self.group_indices = group_indices
        # self.cls_distribution = pd.DataFrame(self.group_indices).value_counts()
        # self.cls_reweighting = {cls_idx[0]: val for cls_idx, val in (1 / (self.cls_distribution / self.cls_distribution.sum())).to_dict().items()}
        # self.cls_reweighting_2 = {cls_idx[0]: val for cls_idx, val in (1 / self.cls_distribution).to_dict().items()}
        # imbsample = RandomOverSampler().fit(self.traces, y=group_indices)
        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")

    def _get_next_activities(self):
        next_line = np.roll(self.data_container, -1, axis=1)
        next_line[:, -1, self.idx_event_attribute] = self.vocab2idx[self.end_token]
        all_next_activities = next_line[:, :, self.idx_event_attribute].astype(int)
        all_next_activities[all_next_activities == self.start_id] = 0
        return all_next_activities

    def viz_dfg(self, bg_color="transparent", save=False):
        start_time = time.time()
        dfg = dfg_discovery.apply(self.log)
        gviz = dfg_visualization.apply(dfg, log=self.log, variant=dfg_visualization.Variants.FREQUENCY)
        gviz.graph_attr["bgcolor"] = bg_color
        self.time_stats["visualize_dfg"] = time.time() - start_time
        if save:
            return dfg_visualization.save(gviz, DATA_FOLDER_VISUALIZATION / (type(self).__name__ + "_dfg.png"))
        return dfg_visualization.view(gviz)

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

    def get_event_attr_len(self, ft_mode: int = FeatureModes.EVENT_ONLY):
        results = self._prepare_input_data(self.trace_train[:5], None, ft_mode)
        return results[0].shape[-1] if not type(results[0]) == tuple else results[0][1].shape[-1]

    # TODO: Change to less complicated output
    def _generate_examples(self, data_mode: int = DatasetModes.TRAIN, ft_mode: int = FeatureModes.EVENT_ONLY) -> Iterator:
        """Generator of examples for each split."""

        data = self._choose_dataset_shard(data_mode)
        res_features, res_targets, res_sample_weights = self._prepare_input_data(*data, ft_mode)
        return res_features, res_targets, res_sample_weights

    def _choose_dataset_shard(self, data_mode):
        if DatasetModes(data_mode) == DatasetModes.TRAIN:
            data = (self.trace_train, self.target_train)
        if DatasetModes(data_mode) == DatasetModes.VAL:
            data = (self.trace_val, self.target_val)
        if DatasetModes(data_mode) == DatasetModes.TEST:
            data = (self.trace_test, self.target_test)
        return data

    def _prepare_input_data(
            self,
            features: np.ndarray,
            targets: np.ndarray = None,
            ft_mode: int = FeatureModes.EVENT_ONLY,
    ) -> tuple:
        res_features = None
        res_targets = None
        res_sample_weights = None
        if ft_mode == FeatureModes.ENCODER_DECODER:
            res_features = features
        if ft_mode == FeatureModes.EVENT_ONLY:
            res_features = features[:, :, self.idx_event_attribute]
        if ft_mode == FeatureModes.EVENT_TIME_SEP:
            res_features = (features[:, :, self.idx_event_attribute], features[:, :, self.idx_time_attributes])
        if ft_mode == FeatureModes.EVENT_TIME:
            res_features = np.concatenate([to_categorical(features[:, :, self.idx_event_attribute]), features[:, :, self.idx_time_attributes]], axis=-1)
        if ft_mode == FeatureModes.FULL_SEP:
            res_features = (features[:, :, self.idx_event_attribute], features[:, :, self.idx_features])
        if ft_mode == FeatureModes.FEATURES_ONLY:
            res_features = features[:, :, self.idx_features]
        if ft_mode == FeatureModes.FULL:
            res_features = np.concatenate([to_categorical(features[:, :, self.idx_event_attribute]), features[:, :, self.idx_features]], axis=-1)
        if ft_mode == FeatureModes.EVENT_ONLY_ONEHOT:
            res_features = to_categorical(features[:, :, self.idx_event_attribute])

        if not ft_mode == FeatureModes.ENCODER_DECODER:
            self.feature_len = res_features.shape[-1] if not type(res_features) == tuple else res_features[1].shape[-1]
        if targets is not None:
            res_sample_weights = self._compute_sample_weights(targets)
            # res_sample_weights = res_sample_weights if res_sample_weights.shape[1] != 1 else res_sample_weights.flatten()

            res_targets = targets 
            # res_targets = res_targets if res_targets.shape[1] != 1 else res_targets.flatten()
            
            return res_features, res_targets, res_sample_weights
        return res_features, None

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     target_counts = Counter(tuple(row) for row in targets)
    #     sum_vals = sum(list(target_counts.values()))
    #     target_weigts = {k: sum_vals / v for k, v in target_counts.items()}
    #     weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
    #     return weighting

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     target_counts = Counter(tuple(row) for row in targets)
    #     target_weigts = {k: 1/v for k, v in target_counts.items()}
    #     weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
    #     return weighting/weighting.sum()

    def _compute_sample_weights(self, targets):
        # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
        # TODO: Default return weight might be tweaked to 1/len(features) or 1
        target_counts = Counter(tuple(row) for row in targets)
        target_weigts = {k: 1/v for k, v in target_counts.items()}
        weighting = np.array([target_weigts.get(tuple(row), 1) for row in targets])[:, None]
        return weighting

    # def _compute_sample_weights(self, targets):
    #     # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
    #     # TODO: Default return weight might be tweaked to 1/len(features) or 1
    #     weighting = np.array([1 for row in targets])[:, None]
    #     return weighting

    def get_dataset(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TRAIN, ft_mode: FeatureModes = FeatureModes.EVENT_ONLY):
        results = self._generate_examples(data_mode, ft_mode)
        if self.mode == TaskModes.ENCODER_DECODER:
            return tf.data.Dataset.from_generator(lambda: results, tf.int64, output_shapes=[None])
        return tf.data.Dataset.from_tensor_slices(results).batch(batch_size)

    def get_dataset_with_indices(self, batch_size=1, data_mode: DatasetModes = DatasetModes.TEST, ft_mode: FeatureModes = FeatureModes.EVENT_ONLY):
        collector = []
        dataset = None
        # dataset = self.get_dataset(1, data_mode, ft_mode)
        trace, target = self._choose_dataset_shard(data_mode)
        res_features, res_targets, res_sample_weights = self._prepare_input_data(trace, target, ft_mode)
        res_indices = trace[:, :, self.idx_event_attribute]
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

    def _heuristic_sample_size(self, sequence):
        return range((len(sequence)**2 + len(sequence)) // 4)

    def _heuristic_bounded_sample_size(self, sequence):
        return range(min((len(sequence)**2 + len(sequence) // 4), 5))

    def _get_example_trace_subset(self, num_traces=10):
        random_starting_point = random.randint(0, self._log_size - num_traces - 1)
        df_traces = pd.DataFrame(self._traces.items()).set_index(0).sort_index()
        example = df_traces[random_starting_point:random_starting_point + num_traces]
        return [val for val in example.values]

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


def test_dataset(reader: AbstractProcessLogReader, batch_size=42, ds_mode: DatasetModes = None, ft_mode: FeatureModes = None):
    def show_instance(reader, batch_size, ds_mode, ft_mode):
        print(f"==================== {ds_mode.name} - {ft_mode.name} ===================")
        data = reader.get_dataset(batch_size, ds_mode, ft_mode)
        data_point = next(iter(data))
        if type(data_point[0]) == tuple:
            print("FEATURES")
            print(data_point[0][0].shape)
            print(data_point[0][1].shape)
        else:
            print("FEATURES")
            print(data_point[0].shape)

        print("TARGET")
        print(data_point[1].shape)
        print(f"=======================================================")

    params = it.product(DatasetModes if ds_mode is None else [ds_mode], FeatureModes if ft_mode is None else [ft_mode])
    for ds_mode, ft_mode in params:
        show_instance(reader, batch_size, ds_mode, ft_mode)


if __name__ == '__main__':
    reader = AbstractProcessLogReader(
        log_path=DATA_FOLDER / 'dataset_bpic2020_tu_travel/RequestForPayment.xes',
        csv_path=DATA_FOLDER_PREPROCESSED / 'RequestForPayment.csv',
        mode=TaskModes.OUTCOME_EXTENSIVE,
    )
    # data = data.init_log(save=0)
    reader = reader.init_data()
    test_dataset(reader)
    print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    features, targets = reader._prepare_input_data(reader.trace_test[0:1], reader.target_test[0:1])
    print(reader.decode_matrix(features[0:1]))
    print(reader.get_data_statistics())