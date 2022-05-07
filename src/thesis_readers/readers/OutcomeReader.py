from collections import Counter
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.helper import test_reader
from thesis_readers.helper.constants import DATA_FOLDER_PREPROCESSED, DATA_FOLDER
from .AbstractProcessLogReader import AbstractProcessLogReader, CSVLogReader, test_dataset
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG

class OutcomeBPIC2011(CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(
            log_path= DATA_FOLDER/ 'dataset_various_outcome_prediction/BPIC11_f1.csv',
            csv_path= DATA_FOLDER_PREPROCESSED/ 'BPIC11_f1.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity code",
            col_timestamp="time:timestamp",
            **kwargs,
        )
        self.col_outcome = "label"

    def gather_information_about_traces(self):
        self.length_distribution = Counter([len(tr) for tr in self._traces.values()])
        self.max_len = max(list(self.length_distribution.keys())) + 2
        self.min_len = min(list(self.length_distribution.keys())) + 2
        self.log_len = len(self._traces)
        self._original_feature_len = len(self.data.columns)
        self.current_feature_len = len(self.data.columns)
        self.idx_event_attribute = self.data.columns.get_loc(self.col_activity_id)
        self.idx_outcome = self.data.columns.get_loc(self.col_outcome)
        self.idx_time_attributes = [self.data.columns.get_loc(col) for col in self.col_timestamp_all]
        self.idx_features = [self.data.columns.get_loc(col) for col in self.data.columns if col not in [self.col_activity_id, self.col_case_id, self.col_timestamp, self.col_outcome]]

    def instantiate_dataset(self, mode: TaskModes = None, add_start: bool = None, add_end: bool = None):
        # TODO: Add option to mirror train and target
        # TODO: Add option to add boundary tags
        print("Preprocess data")
        self.mode = mode or self.mode or TaskModes.NEXT_OUTCOME
        self.data_container = self._put_data_to_container()

        initial_data = np.array(self.data_container)
        all_indices = list(range(initial_data.shape[-1]))
        all_indices.remove(self.idx_outcome)

        if self.mode == TaskModes.OUTCOME_PREDEFINED:
            tmp_data = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
            out_come = initial_data[:, :, self.idx_outcome]  # ATTENTION .reshape(-1)
            self.traces_preprocessed = tmp_data[:, :, all_indices], out_come

        self.traces, self.targets = self.traces_preprocessed
        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")
        return self

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

    def preprocess_level_specialized(self, **kwargs):
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        num_encoder = StandardScaler()

        categorical_columns = list(self.data.select_dtypes('object').columns.drop([self.col_activity_id, self.col_case_id, self.col_outcome]))
        # normalization_columns = list(self.data.select_dtypes('number').columns)
        self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))

        # self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)

    
    
if __name__ == '__main__':
    save_preprocessed = True
    reader = OutcomeBPIC2011(debug=True, mode=TaskModes.OUTCOME_PREDEFINED)
    reader = reader.init_log(save_preprocessed)
    # test_reader(reader, True)

    reader = reader.init_meta()
    test_dataset(reader, 42, ds_mode=DatasetModes.TRAIN, tg_mode=TaskModes.OUTCOME_PREDEFINED, ft_mode=FeatureModes.EVENT_ONLY)
    print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    features, targets, sample_weights = reader._prepare_input_data(reader.trace_test[0:1], reader.target_test[0:1])
    print(reader.decode_matrix(features[0:1]))
    print(reader.get_data_statistics())