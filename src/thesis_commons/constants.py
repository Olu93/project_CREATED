import pathlib
import importlib_resources

from thesis_commons.functions import create_path

PATH_ROOT = importlib_resources.files(__package__).parent.parent
PATH_MODELS = PATH_ROOT / "models"
PATH_MODELS_PREDICTORS = PATH_MODELS / "predictors"
PATH_MODELS_GENERATORS = PATH_MODELS / "generators"

print("================= Folders =====================")
create_path("PATH_ROOT", PATH_ROOT)
create_path("PATH_MODELS", PATH_MODELS)
create_path("PATH_MODELS_PREDICTORS", PATH_MODELS_PREDICTORS)
create_path("PATH_MODELS_GENERATORS", PATH_MODELS_GENERATORS)
print("==============================================")