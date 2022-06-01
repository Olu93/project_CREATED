import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader

from .AbstractProcessLogReader import AbstractProcessLogReader


class HospitalLogReader(AbstractProcessLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path=DATA_FOLDER / 'dataset_bpic2011_hospital_gynocology/hospital_log.xes', csv_path=DATA_FOLDER_PREPROCESSED / 'HospitalTreatment.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[
            "Activity code",
            "lifecycle:transition",
        ], max_diversity_thresh=0.9)  # To agressive for this dataset!

    def preprocess_level_specialized(self, **kwargs):
        # redundant_cols = list(self.data.columns[self.data.columns.str.contains('code:')])
        # all_unimportant_dates = list(self.data.columns[self.data.columns.str.contains('date:')])
        time_columns = list(self.data.select_dtypes('datetimetz').columns.drop(self.col_timestamp))
        self.data = self.data.drop(time_columns, axis=1)
        # categorical_columns = [
        #     "org:group",
        #     "Specialism Code",
        #     "Producer Code",
        #     "Section",
        # ]
        categorical_columns = list(self.data.select_dtypes('object').columns.drop([self.col_activity_id, self.col_case_id]))

        normalization_columns = list(self.data.select_dtypes('number').columns) 

        # cat_encoder = ce.HashingEncoder(verbose=1, return_df=True)
        # cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=1)
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        # cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=3)
        num_encoder = StandardScaler()

        self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))
        self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)


# tmp = {key: AbstractProcessLogReader.gather_grp_column_statsitics(grp_df) for key, grp_df in self._original_data.groupby(self.col_case_id)}
# {key:val for key, val in pd.json_normalize(list(tmp.values())).sum().items() if "entropy" in key}

if __name__ == '__main__':
    reader = HospitalLogReader()
    test_reader(reader, True)