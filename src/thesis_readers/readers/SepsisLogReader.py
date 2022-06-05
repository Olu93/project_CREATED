
from thesis_readers.helper.constants import (DATA_FOLDER,
                                             DATA_FOLDER_PREPROCESSED)
from thesis_readers.helper.helper import test_reader

from .AbstractProcessLogReader import AbstractProcessLogReader


class SepsisLogReader(AbstractProcessLogReader):
    COL_LIFECYCLE = "lifecycle:transition"

    def __init__(self, **kwargs) -> None:
        super().__init__(log_path=DATA_FOLDER / 'dataset_hospital_sepsis/Sepsis Cases - Event Log.xes', csv_path=DATA_FOLDER_PREPROCESSED / 'Sepsis.csv', **kwargs)

    def preprocess_level_general(self):
        super().preprocess_level_general(remove_cols=None)

    def preprocess_level_specialized(self, **kwargs):
        super().preprocess_level_specialized(**kwargs)


if __name__ == '__main__':
    reader = SepsisLogReader()
    test_reader(reader, True)