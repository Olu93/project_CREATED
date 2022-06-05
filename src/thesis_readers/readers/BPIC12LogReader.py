

import category_encoders as ce
from sklearn.preprocessing import StandardScaler

from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader

from .AbstractProcessLogReader import AbstractProcessLogReader


class BPIC12LogReader(AbstractProcessLogReader):
    class subsets:
        A = ('A_')
        W = ('W_')
        O = ('O_')
        AW = ('A_', 'W_')
        AO = ('A_', 'O_')
        WO = ('W_', 'O_')
        AWO = ('A_', 'W_', 'O_')

    def __init__(self, subset=subsets.AW, **kwargs) -> None:
        self.subset = subset
        super().__init__(log_path=DATA_FOLDER / 'dataset_bpic2012_financial_loan/financial_log.xes', csv_path=DATA_FOLDER_PREPROCESSED / 'FinancialLog.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=["org:resource"])

    def preprocess_level_specialized(self, **kwargs):
        self.data = self.data[self.data['lifecycle:transition'] == 'COMPLETE']
        self.data = self.data[self.data[self.col_activity_id].str.startswith(self.subset, na=False)]
        self.data = self.data.drop('lifecycle:transition', axis=1)
        self.data["case:AMOUNT_REQ"] = self.data["case:AMOUNT_REQ"].astype("float") 
        
        cat_encoder = ce.BaseNEncoder(verbose=1, return_df=True, base=2)
        num_encoder = StandardScaler()

        # categorical_columns = list(self.data.select_dtypes('object').columns.drop([self.col_activity_id, self.col_case_id]))
        normalization_columns = list(self.data.select_dtypes('number').columns)
        # self.data = self.data.join(cat_encoder.fit_transform(self.data[categorical_columns]))

        self.data[normalization_columns] = num_encoder.fit_transform(self.data[normalization_columns])
        # self.data = self.data.drop(categorical_columns, axis=1)

        self.preprocessors['categoricals'] = cat_encoder
        self.preprocessors['normalized'] = num_encoder
        super().preprocess_level_specialized(**kwargs)


if __name__ == '__main__':
    reader = BPIC12LogReader()
    print(test_reader(reader, False))
    # reader = reader.init_log(True)
    # reader = reader.init_data()
    # ds_counter = reader.get_dataset()

    # example = next(iter(ds_counter.batch(10)))
    # print(example[0][0].shape)
    # print(example[0][1].shape)
    # reader.viz_bpmn("white")
    # reader.viz_process_map("white")
    # reader.viz_dfg("white")
    # print(reader.get_data_statistics())