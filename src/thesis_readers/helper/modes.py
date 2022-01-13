
from enum import IntEnum, auto, Enum


class TaskModes(Enum):
    OUTCOME = auto()
    NEXT_EVENT = auto()
    OUTCOME_EXTENSIVE = auto()
    NEXT_EVENT_EXTENSIVE = auto()
    # ENCODER_DECODER = auto()
    # EXTENSIVE = auto()
    # EXTENSIVE_RANDOM = auto()
    
class TaskModeType(Enum):
    MANY2MANY = auto()
    MANY2ONE = auto()
    
    @staticmethod
    def type(t:TaskModes):
        if t in [TaskModes.NEXT_EVENT,TaskModes.OUTCOME]:
            return TaskModeType.MANY2ONE
        if t in [TaskModes.NEXT_EVENT_EXTENSIVE,TaskModes.OUTCOME_EXTENSIVE]:
            return TaskModeType.MANY2MANY

class DatasetModes(IntEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class FeatureModes(IntEnum):
    FULL = auto()
    FULL_SEP = auto()
    EVENT_ONLY = auto()
    EVENT_ONLY_ONEHOT = auto()
    FEATURES_ONLY = auto()
    EVENT_TIME = auto()
    EVENT_TIME_SEP = auto()


# class TargetModes(IntEnum):
#     FULL = auto()
#     FULL_SEP = auto()
#     EVENT_ONLY = auto()
#     EVENT_TIME = auto()