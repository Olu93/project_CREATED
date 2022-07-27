from collections import Counter
import pickle
import numpy as np
import pandas as pd
import itertools as it
from pm4py.objects.conversion.log import converter as log_converter

from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.helper.constants import (DATA_FOLDER, DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.preprocessing import BinaryEncodeOperation, CategoryEncodeOperation, ColStats, ComputeColStatsOperation, DropOperation, NumericalEncodeOperation, ProcessingPipeline, Selector, SetIndexOperation
from thesis_readers.readers.MockReader import MockReader
from thesis_commons.constants import CDType
from .AbstractProcessLogReader import CSVLogReader, test_dataset

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG
DEBUG_SHORT_READER_LIMIT = 25

class LimitedMaxLengthReaderMixin():
    def limit_data(self, data:pd.DataFrame, case_id, event_id, limit=None, *args, **kwargs):
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



class OutcomeBPIC12Reader25(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 25)
        return data, {}


class OutcomeBPIC12Reader50(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 50)
        return data, {}
    
class OutcomeBPIC12Reader75(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 75)
        return data, {}
    
class OutcomeBPIC12Reader100(OutcomeBPIC12Reader):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data = self.limit_data(data, self.col_case_id, self.col_activity_id, 100)
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

    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        return data.copy(), {'remove_cols': ['to_drop_at_start']}


class OutcomeDice4ELReader(OutcomeBPIC12Reader50):
    def pre_pipeline(self, data: pd.DataFrame, **kwargs):
        data, super_kwargs = super(OutcomeDice4ELReader, self).pre_pipeline(data, **kwargs)
        df: pd.DataFrame = pickle.load(open('thesis_readers/data/dataset_dice4el/df.pickle', 'rb'))
        keep_cases = df['caseid'].values.astype(int)
        data = data[data[self.col_case_id].isin(keep_cases)]
        data = data[[self.col_case_id, self.col_activity_id, self.col_outcome, 'Resource', 'AMOUNT_REQ', 'event_nr']]
        data = data.sort_values([self.col_case_id, 'event_nr'])
        return data, {'remove_cols': ['event_nr'], **super_kwargs}

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
        return data, {'remove_cols': ['activity_id','resource_id'], **super_kwargs}

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
    save_preprocessed = True
    reader = OutcomeDice4ELEvalReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(True)
    reader.save(True)
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