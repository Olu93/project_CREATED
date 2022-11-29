from enum import Enum
import pathlib
import importlib_resources
from thesis_commons.config import DEBUG_DATASET, READER

from thesis_commons.functions import create_path
import tensorflow as tf

keras = tf.keras
from keras import backend as K
from keras.utils.losses_utils import ReductionV2

PATH_ROOT: pathlib.Path = importlib_resources.files(__package__).parent.parent
PATH_MODELS = PATH_ROOT / "models"
PATH_MODELS_PREDICTORS = PATH_MODELS / "predictors"
PATH_MODELS_GENERATORS = PATH_MODELS / "generators"
PATH_MODELS_OTHERS = PATH_MODELS / "others"
PATH_RESULTS = PATH_ROOT / "results"
PATH_RESULTS_MODELS_OVERALL = PATH_RESULTS / "models_overall"
PATH_RESULTS_MODELS_SPECIFIC = PATH_RESULTS / "models_specific"
PATH_READERS = PATH_ROOT / "readers"
PATH_PAPER = PATH_ROOT / "latex" / 'thesis_phase_2'
PATH_PAPER_FIGURES = PATH_PAPER / "figures/generated"
PATH_PAPER_TABLES = PATH_PAPER / "tables/generated"
PATH_PAPER_COUNTERFACTUALS = PATH_PAPER / "tables/counterfactuals"
# ROOT = pathlib.Path('.')
# PATH_PAPER = ROOT / "latex" / 'thesis_phase_2'
# PATH_PAPER_FIGURES = PATH_PAPER / "figures/generated"
# PATH_PAPER_TABLES = PATH_PAPER / "tables/generated"
print("================= Folders =====================")
create_path("PATH_ROOT", PATH_ROOT)
create_path("PATH_MODELS", PATH_MODELS)
create_path("PATH_MODELS_PREDICTORS", PATH_MODELS_PREDICTORS)
create_path("PATH_MODELS_GENERATORS", PATH_MODELS_GENERATORS)
print("==============================================")


class StringEnum(str, Enum):
    def __repr__(self):
        return self.name


class CMeta(StringEnum):
    IMPRT = 'important'
    FEATS = 'features'
    NON = 'other'


class CDType(StringEnum):
    BIN = 'binaricals'
    CAT = 'categoricals'
    NUM = 'numericals'
    TMP = 'temporals'
    NON = 'other'


class CDomainMappings():
    ALL_DISCRETE = [CDType.BIN, CDType.CAT]
    ALL_CONTINUOUS = [CDType.NUM, CDType.TMP]
    ALL_IMPORTANT = ['inportant', 'timestamp']
    ALL = [CDType.BIN, CDType.CAT, CDType.NUM, CDType.TMP]


class CDomain(StringEnum):
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    NON = 'none'

    @classmethod
    def map_dtype(cls, dtype):
        if dtype in CDomainMappings.ALL_DISCRETE:
            return cls.DISCRETE
        if dtype in CDomainMappings.ALL_CONTINUOUS:
            return cls.CONTINUOUS

        return cls.NON


DS_BPIC_S = 'OutcomeBPIC12Reader25'
DS_BPIC_M = 'OutcomeBPIC12Reader50'
DS_BPIC_L = 'OutcomeBPIC12Reader75'
DS_BPIC_XL = 'OutcomeBPIC12Reader100'
DS_BPIC_XXL = 'OutcomeBPIC12ReaderFull'
DS_LITERATURE = 'OutcomeDice4ELReader'
DS_SEPSIS_S = 'OutcomeSepsisReader25'
DS_SEPSIS_M = 'OutcomeSepsisReader50'
DS_SEPSIS_L = 'OutcomeSepsisReader75'
DS_SEPSIS_XL = 'OutcomeSepsisReader100'
DS_TRAFFIC = 'OutcomeTrafficFineReader'
DS_TRAFFIC_SHORT = 'OutcomeTrafficShortReader'
MAIN_READER = READER
ALL_DATASETS = [
    DS_BPIC_S,
    DS_BPIC_M,
    DS_BPIC_L,
    DS_BPIC_XL,
    DS_SEPSIS_S,
    DS_SEPSIS_M,
    DS_SEPSIS_L,
    DS_SEPSIS_XL,
    DS_TRAFFIC,
    DS_TRAFFIC_SHORT,
    # DS_LITERATURE,
] if not DEBUG_DATASET else [MAIN_READER]

REDUCTION = ReductionV2
