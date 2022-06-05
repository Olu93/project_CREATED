
import importlib_resources

DATA_FOLDER = importlib_resources.files(__package__).parent / "data"
# DATA_FOLDER = importlib_resources.files("src.thesis_data_readers.data")
#  pathlib.Path(__file__).parent.parent / "data"
DATA_FOLDER_PREPROCESSED = DATA_FOLDER / "preprocessed"
DATA_FOLDER_VISUALIZATION = DATA_FOLDER / "graphs"

print("================= Folder =====================")
print(f"Data Folder: {DATA_FOLDER}")
print(f"Preprocessed Data Folder: {DATA_FOLDER_PREPROCESSED}")
print(f"Process Graphs Folder: {DATA_FOLDER_VISUALIZATION}")
print("==============================================")
# print(importlib_resources.files(__package__).parent / "data")
