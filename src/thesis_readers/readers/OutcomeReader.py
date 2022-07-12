from collections import Counter

import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter

from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.readers.MockReader import MockReader

from .AbstractProcessLogReader import CSVLogReader, test_dataset

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG
DEBUG_SHORT_READER_LIMIT = 25

class OutcomeReader(CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.important_cols = self.important_cols.set_col_outcome("label") 


    def _compute_sample_weights(self, targets):
        # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
        # TODO: Default return weight might be tweaked to 1/len(features) or 1
        target_counts = Counter(list(targets.flatten()))  
        sum_vals = sum(list(target_counts.values()))
        target_weigts = {k: sum_vals / v for k, v in target_counts.items()}
        weighting = np.array([target_weigts.get(row, 1) for row in list(targets.flatten())])[:, None]
        return weighting



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

    def preprocess(self, **kwargs):
        return super().preprocess(remove_cols=['event_nr'])


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
    
    def preprocess(self, **kwargs):
        return super().preprocess(remove_cols=['to_drop_at_start'])     

class OutcomeBPIC12ReaderShort(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        seq_counts = data.groupby(self.col_case_id).count()
        keep_cases = seq_counts[seq_counts[self.col_activity_id] <= 25][self.col_activity_id] 
        data = data.set_index(self.col_case_id).loc[keep_cases.index].reset_index()
        return data, {}
    
class OutcomeBPIC12ReaderMedium(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        seq_counts = data.groupby(self.col_case_id).count()
        keep_cases = seq_counts[seq_counts[self.col_activity_id] <= 50][self.col_activity_id] 
        data = data.set_index(self.col_case_id).loc[keep_cases.index].reset_index()
        return data, {}

class OutcomeBPIC12ReaderFull(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        return data, {}

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
        
    def preprocess(self, **kwargs):
        return super().preprocess(remove_cols=['to_drop_at_start'])        
        

if __name__ == '__main__':
    save_preprocessed = True
    reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderShort(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderMedium(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderFull(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)



    # test_dataset(reader, 42, ds_mode=DatasetModes.TRAIN, tg_mode=TaskModes.OUTCOME_PREDEFINED, ft_mode=FeatureModes.FULL)
    # print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    # features, targets = reader._prepare_input_data(reader.trace_test[0:2], reader.target_test[0:2])
    # print(reader.decode_matrix(features[0]))
    # print(reader.get_data_statistics())
    # print("Done!")