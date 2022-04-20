import pathlib
import importlib_resources
from tensorflow.python.keras.utils.losses_utils import ReductionV2
from thesis_commons.functions import create_path

PATH_ROOT = importlib_resources.files(__package__).parent.parent
PATH_MODELS = PATH_ROOT / "models"
PATH_MODELS_PREDICTORS = PATH_MODELS / "predictors"
PATH_MODELS_GENERATORS = PATH_MODELS / "generators"
PATH_MODELS_OTHERS = PATH_MODELS / "others"

print("================= Folders =====================")
create_path("PATH_ROOT", PATH_ROOT)
create_path("PATH_MODELS", PATH_MODELS)
create_path("PATH_MODELS_PREDICTORS", PATH_MODELS_PREDICTORS)
create_path("PATH_MODELS_GENERATORS", PATH_MODELS_GENERATORS)
print("==============================================")

REDUCTION = ReductionV2