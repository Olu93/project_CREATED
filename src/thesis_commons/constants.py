
import pathlib
import importlib_resources
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from thesis_commons.functions import create_path
from thesis_commons.libcuts import K

PATH_ROOT: pathlib.Path = importlib_resources.files(__package__).parent.parent
PATH_MODELS = PATH_ROOT / "models"
PATH_MODELS_PREDICTORS = PATH_MODELS / "predictors"
PATH_MODELS_GENERATORS = PATH_MODELS / "generators"
PATH_MODELS_OTHERS = PATH_MODELS / "others"
PATH_RESULTS = PATH_ROOT / "results"
PATH_RESULTS_MODELS_OVERALL = PATH_RESULTS / "models_overall"
PATH_RESULTS_MODELS_SPECIFIC = PATH_RESULTS / "models_specific"
PATH_READERS = PATH_ROOT / "readers"

print("================= Folders =====================")
create_path("PATH_ROOT", PATH_ROOT)
create_path("PATH_MODELS", PATH_MODELS)
create_path("PATH_MODELS_PREDICTORS", PATH_MODELS_PREDICTORS)
create_path("PATH_MODELS_GENERATORS", PATH_MODELS_GENERATORS)
print("==============================================")

class CustomReduction(ReductionV2):
    AUTO = 'auto'
    NONE = 'none'
    ALL_SUM = 'sum'
    ALL_AVG = 'avg'
    SEQ_AVG = 'avg_over_sequence' 
    SEQ_SUM = 'sum_over_sequence' 
    SEQ_AVG_OVER_BATCH = 'avg_over_sequence_over_batch' 
    SEQ_SUM_OVER_BATCH = 'sum_over_sequence_over_batch' 
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'    

    @staticmethod
    def reduce_result(reduction, values):
        if reduction == CustomReduction.NONE:
            return values
        if reduction == CustomReduction.SEQ_SUM:
            sum_over_sequence = K.sum(values, axis=-1)
            sequence_sum = K.sum(values)
            return K.sum(K.sum(sum_over_sequence / sequence_sum))
        if reduction == CustomReduction.SEQ_AVG:
            sum_over_sequence = K.sum(values, axis=-1)
            sequence_sum = K.sum(values)
            return K.mean(K.sum(sum_over_sequence / sequence_sum))
        if reduction == CustomReduction.ALL_SUM:
            return K.sum(values)
        if reduction == CustomReduction.ALL_AVG:
            return K.mean(values)


    # def __init__(self, reduction=None):
    #     super().__init__()

    # def reduce(self, values):
    #     if 
REDUCTION = ReductionV2 