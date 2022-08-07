from __future__ import annotations
from collections import Counter
from enum import IntEnum, auto
import pickle
import numpy as np
import pandas as pd
import itertools as it
from pm4py.objects.conversion.log import converter as log_converter
from thesis_commons.functions import reverse_sequence_2

from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER, DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.preprocessing import BinaryEncodeOperation, CategoryEncodeOperation, ColStats, ComputeColStatsOperation, DropOperation, NumericalEncodeOperation, ProcessingPipeline, Selector, SetIndexOperation
from thesis_readers.readers.MockReader import MockReader
from thesis_commons.constants import CDType
from .AbstractProcessLogReader import CSVLogReader, test_dataset
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss, RandomUnderSampler

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG
DEBUG_SHORT_READER_LIMIT = 25


class LimitedMaxLengthReaderMixin():
    def limit_data(self, data: pd.DataFrame, case_id, event_id, limit=None, *args, **kwargs):
        if not limit:
            return data
        seq_counts = data.groupby(case_id).count()
        keep_cases = seq_counts[seq_counts[event_id] <= limit][event_id]
        data = data.set_index(case_id).loc[keep_cases.index].reset_index()
        self._virtual_max_len = limit
        return data

    @property
    def virual_max_len(self):
        if not hasattr(self, "_virtual_max_len"):
            return None
        return self._virtual_max_len + 2


class OutcomeReader(LimitedMaxLengthReaderMixin, CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.important_cols = self.important_cols.set_col_outcome("label")
        self._vocab_outcome = {'regular':0, 'deviant':1}

    def _compute_sample_weights(self, targets):
        # mask = np.not_equal(targets, 0) & np.not_equal(targets, self.end_id)
        # TODO: Default return weight might be tweaked to 1/len(features) or 1
        target_counts = Counter(list(targets.flatten()))
        sum_vals = sum(list(target_counts.values()))
        target_weigts = {k: sum_vals / v for k, v in target_counts.items()}
        weighting = np.array([target_weigts.get(row, 1) for row in list(targets.flatten())])[:, None]
        return weighting

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
        X, Y, indices, undersample = self.balance_data(features_container, target_container, self.idx_event)
        self.traces_preprocessed = X, Y
        self.traces, self.targets = self.traces_preprocessed

        self.trace_data, self.trace_test, self.target_data, self.target_test = train_test_split(self.traces, self.targets)

        self.trace_train, self.trace_val, self.target_train, self.target_val = train_test_split(self.trace_data, self.target_data)
        # undersample = NearMiss(version=1, n_neighbors=3, n_jobs=6)
        print(f"All: {len(self.traces)} datapoints")
        print(f"Test: {len(self.trace_test)} datapoints")
        print(f"Train: {len(self.trace_train)} datapoints")
        print(f"Val: {len(self.trace_val)} datapoints")
        return self

    def balance_data(self, X, Y, idx_event, percentile=None):
        undersample = RandomUnderSampler(replacement=False)
        len_train, max_len, num_feature = X.shape
        flat_X = X.reshape((-1, max_len * num_feature))
        flat_Y = Y
        lbl_Y = flat_Y[:, 0]
        if percentile is not None:
            len_dist = (X[:, :, idx_event] != 0).sum(-1)
            len_quantile = np.percentile(len_dist, percentile)
            lbl_Y = [f"{el}" if (el[0] < len_quantile) else f"rare_{el[1]}" for el in zip(len_dist, lbl_Y)]
        
        tmp_X, tmp_Y = undersample.fit_resample(flat_X, lbl_Y[:, None])
        indices = undersample.sample_indices_
        X = flat_X[indices].reshape((-1, max_len, num_feature))
        Y = flat_Y[indices]
        return X, Y, indices, undersample

    def _preprocess_containers(self, mode, add_start, add_end, initial_data):
        if mode == TaskModes.NEXT_OUTCOME:  #_SUPER
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
            all_next_activities = self._get_events_only(features_container, OutcomeReader.shift_mode.NONE)

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
            X, Y = features_container, target_container

        if mode == TaskModes.OUTCOME_PREDEFINED:
            print(f"Normal Mode {self.mode}")
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            target_container = np.max(initial_data[:, :, self.idx_outcome], axis=-1)[..., None]
            X, Y = features_container, target_container

        if mode == TaskModes.OUTCOME_EXTENSIVE_DEPRECATED:
            print(f"Extensive Mode {self.mode}")
            mask = np.zeros((self.max_len, self.max_len))
            for i in range(1, self.max_len + 1):
                mask[i - 1:, :i] = True
            mask = mask.T[None, :, None]
            features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, True if not add_end else add_end)
            self.original_container = np.array(features_container), np.max(initial_data[:, :, self.idx_outcome], axis=-1)[..., None]
            important_shape = features_container.shape[1:]
            reversed_flat_ft_tmp = reverse_sequence_2(features_container)
            ft_tmp = mask * reversed_flat_ft_tmp[..., None]
            flat_ft_tmp = ft_tmp.transpose((0, 3, 1, 2)).reshape((-1, *important_shape))
            useful = ((flat_ft_tmp != 0).sum((-1, -2)) > 1)
            usefule_ft = flat_ft_tmp[useful]  #.reshape((-1, *important_shape))
            usefule_ft = reverse_sequence_2(usefule_ft)
            target_container = np.max(usefule_ft[:, :, self.idx_outcome], axis=-1)[..., None]
            X, Y = usefule_ft, target_container

        # if mode == TaskModes.OUTCOME_EXTENSIVE_DEPRECATED:
        #     # TODO: Design features like next event
        #     features_container = self._add_boundary_tag(initial_data, True if not add_start else add_start, False if not add_end else add_end)
        #     all_next_activities = self._get_events_only(features_container, self.shift_mode.NEXT)

        #     mask = np.not_equal(features_container[:, :, self.idx_event], 0)
        #     target_container = all_next_activities[:, -1][:, None]
        #     extensive_out_come = mask * target_container

        return X, Y


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


class OutcomeSepsisReader(OutcomeReader):
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

    def pre_pipeline(self, data, *args, **kwargs):
        return data, {'remove_cols': ['event_nr']}


class OutcomeSepsisReader25(OutcomeSepsisReader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 25)
        return data, {'remove_cols': ['event_nr']}


class OutcomeSepsisReader50(OutcomeSepsisReader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 50)
        return data, {'remove_cols': ['event_nr']}


class OutcomeSepsisReader75(OutcomeSepsisReader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 75)
        return data, {'remove_cols': ['event_nr']}


class OutcomeSepsisReader100(OutcomeSepsisReader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 100)
        return data, {'remove_cols': ['event_nr']}


class OutcomeBPIC12Reader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super().__init__(
            log_path=DATA_FOLDER / 'dataset_various_outcome_prediction/bpic2012_O_ACCEPTED-COMPLETE.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / f'{type(self).__name__}.csv',
            sep=";",
            col_case_id="Case ID",
            col_event_id="Activity",
            col_timestamp="Complete Timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )

    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = data[data["lifecycle:transition"] == "COMPLETE"]
        data[self.col_activity_id] = data[self.col_activity_id].str.replace("-COMPLETE", "")
        return data, kwargs


class OutcomeBPIC12Reader25(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, kwargs = super().pre_pipeline(data, **kwargs)
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 25)
        return data, kwargs


class OutcomeBPIC12Reader50(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, kwargs = super().pre_pipeline(data, **kwargs)
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 50)
        return data, kwargs


class OutcomeBPIC12Reader75(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, kwargs = super().pre_pipeline(data, **kwargs)
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 75)
        return data, kwargs


class OutcomeBPIC12Reader100(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, kwargs = super().pre_pipeline(data, **kwargs)
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 100)
        return data, kwargs


class OutcomeBPIC12ReaderFull(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, kwargs = super().pre_pipeline(data, **kwargs)
        return data, kwargs


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

    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        return data.copy(), {'remove_cols': ['to_drop_at_start']}


class OutcomeDice4ELReader(OutcomeBPIC12Reader25):
    pad_token: str = "<PAD>"
    end_token: str = "<EOS>"
    start_token: str = "<SOS>" 
    

    def init_meta(self, skip_dynamics: bool = False) -> OutcomeDice4ELReader:
        super().init_meta(skip_dynamics)
        return self.load_d4el_data()
    
    class FormatModes(IntEnum):
        D4ELF = auto() 
        CASEF = auto() 
        DFRAM = auto() 
        NUMPY = auto()        
    
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
    
        data, kwargs = super(OutcomeDice4ELReader, self).pre_pipeline(data, **kwargs)
        df: pd.DataFrame = pickle.load(open('thesis_readers/data/dataset_dice4el/df.pickle', 'rb'))
        keep_cases = df['caseid'].values.astype(int)
        data = data[data[self.col_case_id].isin(keep_cases)]
        data = data[[self.col_case_id, self.col_activity_id, self.col_outcome, 'Resource', 'AMOUNT_REQ', 'event_nr']]
        data = data.sort_values([self.col_case_id, 'event_nr'])
        return data, {'remove_cols': ['event_nr'], **kwargs}

    def construct_pipeline(self, **kwargs):
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
        op = op.chain(SetIndexOperation(name="set_index", digest_fn=Selector.select_static, cols=[self.important_cols.col_case_id]))
        # op = op.append_next(TemporalEncodeOperation(name=CDType.TMP, digest_fn=Selector.select_timestamps))
        op = op.append_next(BinaryEncodeOperation(name=CDType.BIN, digest_fn=Selector.select_binaricals))
        op = op.append_next(CategoryEncodeOperation(name=CDType.CAT, digest_fn=Selector.select_categoricals))
        op = op.append_next(NumericalEncodeOperation(name=CDType.NUM, digest_fn=Selector.select_numericals))

        return pipeline
    
    def load_d4el_data(self, selection = []) -> OutcomeDice4ELReader:
        list_of_cfs = []
        list_of_fas = []
        data = self.original_data
        selection = ['173844', '173847', '173859', '173868', '173871', '173913'] if not len(selection) else selection
        for el in selection:
            cf_data: pd.DataFrame = pd.read_csv(f'thesis_readers/data/dataset_dice4el/d4e_{el}.csv')
            keep_cases = cf_data['caseid_seed'].values.astype(int)
            fa_data = data[data[self.col_case_id].isin(keep_cases)]  
            list_of_cfs.append(cf_data)
            list_of_fas.append(fa_data)
                  
        self.d4e_fas = pd.concat(list_of_fas)
        self.d4e_fas["lifecycle:transition"] = "COMPLETE"
        self.d4e_cfs = pd.concat(list_of_cfs).reset_index(drop=True)
        print("Succesful load of D4EL data")
        return self  
        
    def get_d4el_factuals(self, mode: FormatModes = FormatModes.NUMPY):
        # if mode == self.FormatModes.NUMPY:
        
        (X_events, X_features), Y =  self.encode(self.d4e_fas, TaskModes.OUTCOME_PREDEFINED)
            
        return (X_events, X_features), Y 
    

    
    def get_d4el_counterfactuals(self):
        pass

class OutcomeDice4ELEvalReader(OutcomeReader):
    def __init__(self, **kwargs) -> None:

        super(OutcomeDice4ELEvalReader, self).__init__(
            log_path=DATA_FOLDER / 'dataset_dice4el/labelled_df.csv',
            csv_path=DATA_FOLDER_PREPROCESSED / f'{type(self).__name__}.csv',
            sep=",",
            col_case_id="caseid",
            col_event_id="activity",
            col_timestamp="Complete Timestamp",
            mode=kwargs.pop('mode', TaskModes.OUTCOME_PREDEFINED),
            **kwargs,
        )

    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, super_kwargs = super(OutcomeDice4ELEvalReader, self).pre_pipeline(data, **kwargs)
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 25)
        data = data.drop(['Unnamed: 0', 'pos'], axis=1)
        data = data[~data[self.col_activity_id].str.startswith("<")]
        data = data.groupby(self.col_case_id).head(-1)
        self.keep_data = data.groupby(self.col_case_id).tail(1)
        return data, {'remove_cols': ['activity_id', 'resource_id'], **super_kwargs}

    def construct_pipeline(self, **kwargs):
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
        op = op.chain(SetIndexOperation(name="set_index", digest_fn=Selector.select_static, cols=[self.important_cols.col_case_id]))
        # op = op.append_next(TemporalEncodeOperation(name=CDType.TMP, digest_fn=Selector.select_timestamps))
        op = op.append_next(BinaryEncodeOperation(name=CDType.BIN, digest_fn=Selector.select_binaricals))
        op = op.append_next(CategoryEncodeOperation(name=CDType.CAT, digest_fn=Selector.select_categoricals))
        op = op.append_next(NumericalEncodeOperation(name=CDType.NUM, digest_fn=Selector.select_numericals))

        return pipeline


if __name__ == '__main__':
    # TODO: Put debug stuff into configs
    task_mode = TaskModes.OUTCOME_PREDEFINED
    # save_preprocessed = True
    # reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_EXTENSIVE_DEPRECATED).init_log(save_preprocessed).init_meta(True)
    # reader.save(True)
    save_preprocessed = True
    reader = OutcomeDice4ELReader(debug=True, mode=task_mode).init_log(save_preprocessed).init_meta(False)
    test = reader.original_data.loc[reader.original_data[reader.col_case_id].isin([173688, 173844])]
    test['lifecycle:transition'] = "COMPLETE"
    print(reader.encode(test, task_mode))
    # reader.save(True)
    # reader = OutcomeDice4ELReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(True)
    # reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(True)
    # reader = OutcomeBPIC12Reader25(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(True)
    # reader = OutcomeBPIC12ReaderMedium(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(True)
    # reader = OutcomeBPIC12ReaderFull(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(True)

    # test_dataset(reader, 42, ds_mode=DatasetModes.TRAIN, tg_mode=TaskModes.OUTCOME_PREDEFINED, ft_mode=FeatureModes.FULL)
    # print(reader.prepare_input(reader.trace_test[0:1], reader.target_test[0:1]))

    # features, targets = reader._prepare_input_data(reader.trace_test[0:2], reader.target_test[0:2])
    # print(reader.decode_matrix(features[0]))
    # print(reader.get_data_statistics())
    # print("Done!")