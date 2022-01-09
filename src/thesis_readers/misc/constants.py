import pathlib
import importlib_resources

DATA_FOLDER = importlib_resources.files(__package__).parent / "data"
# DATA_FOLDER = importlib_resources.files("src.thesis_data_readers.data")
#  pathlib.Path(__file__).parent.parent / "data"
DATA_FOLDER_PREPROCESSED = DATA_FOLDER / "preprocessed"
DATA_FOLDER_VISUALIZATION = DATA_FOLDER / "graphs"

print("======================================")
print(DATA_FOLDER)
# print(importlib_resources.files(__package__).parent / "data")
