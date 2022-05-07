from thesis_readers.helper.helper import test_reader
from thesis_readers.helper.constants import DATA_FOLDER_PREPROCESSED, DATA_FOLDER
from .AbstractProcessLogReader import AbstractProcessLogReader, CSVLogReader
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TO_EVENT_LOG = log_converter.Variants.TO_EVENT_LOG

class OutcomeBPIC2011(CSVLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(
            log_path= DATA_FOLDER/ 'dataset_various_outcome_prediction/BPIC11_f1.csv',
            csv_path= DATA_FOLDER_PREPROCESSED/ 'RabobankItil.csv',
            sep=";",
            col_case_id="Incident ID",
            col_event_id="IncidentActivity_Type",
            col_timestamp="DateStamp",
            **kwargs,
        )

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

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
    reader = OutcomeBPIC2011(debug=True)
    test_reader(reader, True)

