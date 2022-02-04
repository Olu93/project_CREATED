from enum import IntEnum, auto, Enum
# TODO: Put into thesis_commons package

class TaskModes(Enum):
    OUTCOME = auto()
    NEXT_EVENT = auto()
    PREV_EVENT = auto()
    OUTCOME_EXTENSIVE_DEPRECATED = auto()
    NEXT_OUTCOME = auto()
    NEXT_EVENT_EXTENSIVE = auto()
    ENCODER_DECODER = auto()
    ENCDEC_EXTENSIVE = auto()
    # EXTENSIVE = auto()
    # EXTENSIVE_RANDOM = auto()


class FeatureModes(IntEnum):
    FULL = auto()
    FULL_SEP = auto()
    EVENT_ONLY = auto()
    EVENT_ONLY_ONEHOT = auto()
    FEATURES_ONLY = auto()
    EVENT_TIME = auto()
    EVENT_TIME_SEP = auto()
    ENCODER_DECODER = auto()


class DatasetModes(IntEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class TaskModeType(Enum):
    FIX2FIX = auto()
    FIX2ONE = auto()
    MANY2MANY = auto()
    MANY2ONE = auto()

    @staticmethod
    def type(t: TaskModes):
        if t in [TaskModes.NEXT_EVENT, TaskModes.OUTCOME, TaskModes.NEXT_OUTCOME, TaskModes.PREV_EVENT]:
            return TaskModeType.FIX2ONE
        if t in [TaskModes.NEXT_EVENT_EXTENSIVE, TaskModes.OUTCOME_EXTENSIVE_DEPRECATED, TaskModes.ENCDEC_EXTENSIVE]:
            return TaskModeType.FIX2FIX
    

class InputModeType(Enum):
    TOKEN_INPUT = auto()
    DUAL_INPUT = auto()
    VECTOR_INPUT = auto()

    @staticmethod
    def type(t: FeatureModes):
        if t in [FeatureModes.EVENT_ONLY]:
            return InputModeType.TOKEN_INPUT
        if t in [FeatureModes.EVENT_TIME_SEP, FeatureModes.FULL_SEP]:
            return InputModeType.DUAL_INPUT
        if t in [FeatureModes.EVENT_ONLY_ONEHOT, FeatureModes.FEATURES_ONLY, FeatureModes.EVENT_TIME, FeatureModes.FULL]:
            return InputModeType.VECTOR_INPUT