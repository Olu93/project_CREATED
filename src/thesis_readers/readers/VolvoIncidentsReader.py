import category_encoders as ce
from sklearn.preprocessing import StandardScaler

from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader

from .AbstractProcessLogReader import AbstractProcessLogReader


class VolvoIncidentsReader(AbstractProcessLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path=DATA_FOLDER / 'dataset_bpic2013_volvo_incidents/bpi_challenge_2013_incidents.xes',
                         csv_path=DATA_FOLDER_PREPROCESSED / 'VolvoIncidents.csv',
                         **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=["org:resource"])

    def preprocess_level_specialized(self, **kwargs):
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        num_encoder = StandardScaler()

        categorical_columns = list(self.data.select_dtypes('object').columns.drop([self.col_activity_id, self.col_case_id]))
        # normalization_columns = list(self.data.select_dtypes('number').columns)
        self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))

        # self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)


if __name__ == '__main__':
    reader = VolvoIncidentsReader()
    test_reader(reader, True, save_viz=True)