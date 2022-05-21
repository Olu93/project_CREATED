from datetime import datetime, timedelta
from typing import Counter
import tensorflow as tf
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from .AbstractProcessLogReader import AbstractProcessLogReader
import numpy as np
import pandas as pd



class MockReader(AbstractProcessLogReader):
    
    def __init__(self, random_feature_len = 9, mode=TaskModes.NEXT_EVENT_EXTENSIVE, debug=False) -> None:
        self.preprocessors = {}
        self.ngram_order = 2
        feature_len = random_feature_len
        self.mode = mode
        self.y_true = np.array([
            [1, 2, 3, 6, 0, 0],
            [1, 2, 1, 2, 3, 4],
            [1, 7, 1, 1, 2, 3],
            [1, 2, 1, 2, 3, 4],
            [1, 7, 1, 1, 2, 3],
            [1, 2, 1, 2, 3, 4],
            [1, 1, 2, 5, 0, 0],
            [7, 1, 2, 5, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 1, 1, 2, 3, 4],
            [1, 1, 2, 5, 0, 0],
            [7, 1, 2, 5, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 1, 1, 2, 3, 4],
            [1, 7, 1, 1, 2, 3],
            [1, 2, 1, 2, 3, 4],
            [1, 1, 2, 5, 0, 0],
            [7, 1, 2, 5, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 1, 1, 2, 3, 4],
            [1, 1, 1, 2, 3, 4],
            [1, 2, 3, 6, 0, 0],
            [1, 3, 2, 6, 0, 0],
            [1, 3, 2, 6, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 3, 2, 6, 0, 0],
            [1, 2, 1, 1, 2, 3],
            [7, 2, 3, 6, 0, 0],
            [1, 2, 1, 2, 3, 4],
            [1, 7, 1, 1, 2, 3],
            [1, 2, 1, 2, 3, 4],
            [7, 1, 2, 5, 0, 0],
            [1, 1, 2, 5, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 2, 1, 2, 3, 4],
            [7, 1, 2, 5, 0, 0],
            [1, 1, 2, 5, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [7, 1, 3, 2, 3, 4],
            [1, 1, 1, 2, 3, 4],
            [1, 2, 3, 6, 0, 0],
            [1, 3, 2, 6, 0, 0],
            [1, 2, 3, 6, 0, 0],
            [1, 3, 2, 6, 0, 0],
            [1, 2, 1, 7, 2, 3],
        ], dtype=np.int32)
        nonzeros = np.nonzero(self.y_true)
        log_len, num_max_events = self.y_true.shape[0], self.y_true.shape[1]
        case_ids = np.arange(1, log_len+1)[:,None] * np.ones_like(self.y_true)
        today = datetime.now()
        log_days = np.random.randint(0,10, size=(log_len, 1))
        days_offsets = np.repeat(np.array(range(0, num_max_events))[None], log_len, axis=0)
        days_offsets = np.cumsum(days_offsets + np.random.randint(1,4, size=self.y_true.shape), axis=1)
        days_offsets = (log_days + days_offsets)
        times = np.array([[today + timedelta(int(offset)) for offset in offset_row] for offset_row in days_offsets])
        features = np.random.uniform(-5, 5, size=(log_len, num_max_events, feature_len))

        ys = self.y_true[nonzeros][None].T
        ids = case_ids[nonzeros][None].T
        tm = times[nonzeros][None].T
        fts = features[nonzeros]
        self._original_data = pd.DataFrame(np.concatenate([ids, tm, fts, ys], axis=1))

        self.col_timestamp = "tm"
        self.col_case_id = "case_id"
        self.col_activity_id = "event_id"
        new_columns = [f"ft_{idx}" for idx in self._original_data.columns]
        new_columns[0] = self.col_case_id
        new_columns[1] = self.col_timestamp
        new_columns[len(self._original_data.columns)-1] = self.col_activity_id
        self._original_data.columns = new_columns
        self._original_data[self.col_activity_id] = self._original_data[self.col_activity_id].astype(str)
        self._original_data[self.col_activity_id] = "activity_" + self._original_data[self.col_activity_id]
        if TaskModes.OUTCOME_PREDEFINED:
            self.col_outcome = "label"
            self._original_data[self.col_outcome] = np.random.random(len(self._original_data)) > 0.66
            self._original_data[self.col_outcome] = self._original_data[self.col_outcome].astype(object)
            self._original_data.loc[self._original_data[self.col_outcome] == False, [self.col_outcome]] =  "regular"
            self._original_data.loc[self._original_data[self.col_outcome] == True, [self.col_outcome]] =  "deviant"
        self._original_data
        self.data = self._original_data
        print("USING MOCK DATASET!")

    def init_meta(self):
        return super().init_meta()

    def init_log(self, save=False):
        self.log = None
        self._original_data = self._original_data
        return self

    def init_data(self):
        self.register_vocabulary()
        self.group_rows_into_traces()
        self.gather_information_about_traces()
        self.instantiate_dataset()
        return self


    def get_dataset(self, batch_size=None, data_mode: DatasetModes = DatasetModes.TRAIN, ft_mode: FeatureModes = FeatureModes.EVENT_ONLY):
        results = self._prepare_input_data(self.traces, self.targets, ft_mode)
        bs = self.log_len if batch_size is None else min([batch_size, self.log_len]) 
        return tf.data.Dataset.from_tensor_slices(results).batch(bs)