from collections import Counter

import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from sklearn.model_selection import train_test_split

from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader
from thesis_readers.readers.MockReader import MockReader

from .AbstractProcessLogReader import (AbstractProcessLogReader, CSVLogReader,
                                       test_dataset)

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG


class OutcomeReader(CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
        self.idx_features = [
            self.data.columns.get_loc(col) for col in self.data.columns if col not in [self.col_activity_id, self.col_case_id, self.col_timestamp, self.col_outcome]
        ]
        
    # def _put_outcome_to_container(self):
    #     data_container = np.zeros([self.log_len, self._original_feature_len])

    #     for idx, (case_id, df) in enumerate(self._traces.items()):

    #         data_container[idx] = df[self.outcome]
    #     return data_container
    
    # def instantiate_dataset(self, mode: TaskModes = None, add_start: bool = None, add_end: bool = None):
    #     # TODO: Add option to mirror train and target
    #     # TODO: Add option to add boundary tags
    #     print("Preprocess data")
    #     self.mode = mode or self.mode or TaskModes.OUTCOME_PREDEFINED
    #     self.data_container = self._put_data_to_container()

    #     initial_data = np.array(self.data_container)
    #     # all_indices = list(range(initial_data.shape[-1]))
    #     # all_indices.remove(self.idx_outcome)

    #     if self.mode == TaskModes.OUTCOME_PREDEFINED:
    #         tmp_data = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
    #         out_come = np.max(initial_data[:, :, self.idx_outcome], axis=-1)[..., None]
    #         self.traces_preprocessed = tmp_data, out_come

    #     self.traces, self.targets = self.traces_preprocessed
    #     self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)
    #     self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
    #     print(f"Test: {len(self.trace_test)} datapoints")
    #     print(f"Train: {len(self.trace_train)} datapoints")
    #     print(f"Val: {len(self.trace_val)} datapoints")
    #     return self

    def _compute_sample_weights(self, targets):
        # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
        # TODO: Default return weight might be tweaked to 1/len(features) or 1
        target_counts = Counter(list(targets.flatten()))  
        sum_vals = sum(list(target_counts.values()))
        target_weigts = {k: sum_vals / v for k, v in target_counts.items()}
        weighting = np.array([target_weigts.get(row, 1) for row in list(targets.flatten())])[:, None]
        return weighting

    def preprocess_level_general(self, remove_cols=None):
        self.data = self.original_data.drop(self.col_timestamp, axis=1)
        return super().preprocess_data(self.data)

    def preprocess_level_specialized(self, **kwargs):
        pass

    def phase_3_time_extract(self, data: pd.DataFrame, col_timestamp=None):
        return data, []

class OutcomeBPIC2011Reader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/BPIC11_f1.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'BPIC11_f1.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity code",
            col_timestamp="time:timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )


class OutcomeProductionReader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/Production.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'production_process.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity",
            col_timestamp="Complete Timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )


class OutcomeTrafficFineReader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/traffic_fines_1.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'traffic_fine_process.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity",
            col_timestamp="Complete Timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )



class OutcomeSepsis1Reader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/sepsis_cases_1.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'sepsis_1.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity",
            col_timestamp="time:timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )

    
    def phase_1_premature_drop(self, data: pd.DataFrame, cols=None):
        cols = ['event_nr']
        new_data = data.drop(cols, axis=1)
        removed_cols = set(data.columns) - set(cols)
        return new_data, removed_cols


class OutcomeBPIC12Reader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/bpic2012_O_ACCEPTED-COMPLETE.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'bpic12_o_accepted.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity",
            col_timestamp="Complete Timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )

    
    def phase_1_premature_drop(self, data: pd.DataFrame, cols=None):
        cols = ['event_nr']
        new_data = data.drop(cols, axis=1)
        removed_cols = set(data.columns) - set(cols)
        return new_data, removed_cols

class OutcomeMockReader(OutcomeReader):
    def __init__(self, **kwargs) -> None:
        reader = MockReader(mode=TaskModes.OUTCOME_PREDEFINED)
        reader._original_data.to_csv(DATA_FOLDER / 'dataset_various_outcome_prediction/mock_data.csv', index=False)
        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/mock_data.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / 'mock_data.csv',
            sep=",",
            col_case_id="case_id",
            col_event_id="event_id",
            col_timestamp="tm",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )
    def preprocess_level_general(self, remove_cols=None):
        self.data = self.original_data
        return super().preprocess_data(self.data)
    
    def phase_3_time_extract(self, data: pd.DataFrame, col_timestamp=None):
        return super(OutcomeReader, self).phase_3_time_extract(data, col_timestamp)

if __name__ == '__main__':
    save_preprocessed = True
    reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader = MockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta()
    # test_reader(reader, True)

    test_dataset(reader, 42, ds_mode=DatasetModes.TRAIN, tg_mode=TaskModes.OUTCOME_PREDEFINED, ft_mode=FeatureModes.FULL)
    print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    features, targets = reader._prepare_input_data(reader.trace_test[0:2], reader.target_test[0:2])
    print(reader.decode_matrix(features[0]))
    print(reader.get_data_statistics())