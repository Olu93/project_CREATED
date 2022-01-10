from thesis_readers.misc.helper import test_reader
from thesis_readers.misc.constants import DATA_FOLDER_PREPROCESSED, DATA_FOLDER
from .AbstractProcessLogReader import AbstractProcessLogReader
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DomesticDeclarationsLogReader(AbstractProcessLogReader):
    COL_DECLARATION_NUM = "case:DeclarationNumber"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path=DATA_FOLDER / 'dataset_bpic2020_tu_travel/DomesticDeclarations.xes', csv_path=DATA_FOLDER_PREPROCESSED / 'DomesticDeclarations.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=[])

    def preprocess_level_specialized(self, **kwargs):
        self.data[self.col_activity_id] = self.data[self.col_activity_id].replace(
            'Declaration ',
            'DECL. ',
            regex=True,
        ).replace(
            ' by ',
            ' ',
            regex=True,
        ).replace(
            ' ',
            '_',
            regex=True,
        )
        self.data[self.col_case_id] = self.data[self.col_case_id].replace(
            ' ',
            '_',
            regex=True,
        )
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        num_encoder = StandardScaler()

        categorical_columns = list(self.data.select_dtypes('object').columns.drop([self.col_activity_id, self.col_case_id]))
        normalization_columns = list(self.data.select_dtypes('number').columns)
        self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))

        self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)


if __name__ == '__main__':
    reader = DomesticDeclarationsLogReader()
    print(test_reader(reader, True))
