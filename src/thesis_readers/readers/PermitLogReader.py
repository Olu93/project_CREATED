import category_encoders as ce
import numpy as np
from sklearn.preprocessing import StandardScaler

from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader

from .AbstractProcessLogReader import AbstractProcessLogReader


class PermitLogReader(AbstractProcessLogReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(log_path=DATA_FOLDER / 'dataset_bpic2020_tu_travel/PermitLog.xes', csv_path=DATA_FOLDER_PREPROCESSED / 'PermitLog.csv', **kwargs)

    def preprocess_level_general(self):

        super().preprocess_level_general(remove_cols=[
            'case:ProjectNumber',
            'case:TaskNumber',
            'case:ActivityNumber',
            'case:Activity_0',
            'case:dec_id_0',
            'case:id',
            'case:travel permit number',
            'case:DeclarationNumber_0',
            'case:OrganizationalEntity_0',
            'case:BudgetNumber',
            'case:Rfp_id_0',
            'case:Task_0',
            'case:Project_0',
            "case:Cost Type_0",
        ])

    def preprocess_level_specialized(self, **kwargs):

        cat_cols = self.data.select_dtypes('object').columns

        cat_cols_with_amount = cat_cols[cat_cols.str.contains('amount', case=False)]
        self.data[cat_cols_with_amount] = self.data[cat_cols_with_amount].replace(r'^\s*$', np.nan, regex=True)
        self.data[cat_cols_with_amount] = self.data[cat_cols_with_amount].stack().str.replace(',', '.').unstack().astype(float)
        # TODO: Cover case for binary encoder with third type
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        num_encoder = StandardScaler()

        categorical_columns = list(cat_cols.drop([self.col_activity_id, self.col_case_id] + list(cat_cols_with_amount)))
        normalization_columns = list(self.data.select_dtypes('number').columns)
        self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))
        self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)


if __name__ == '__main__':
    reader = PermitLogReader()
    test_reader(reader, True)