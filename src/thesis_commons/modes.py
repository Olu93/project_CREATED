from enum import IntEnum, auto, Enum
# TODO: Put into thesis_commons package


class TaskModes(Enum):
    OUTCOME = auto()
    OUTCOME_PREDEFINED = auto()
    NEXT_EVENT = auto()
    OUTCOME_EXTENSIVE_DEPRECATED = auto()
    NEXT_OUTCOME = auto()
    NEXT_EVENT_EXTENSIVE = auto()
    ENCODER_DECODER = auto()
    # EXTENSIVE = auto()
    # EXTENSIVE_RANDOM = auto()

# TODO: Purge feature modes and introduce TIME, FEATURE and FULL. All being in hybrid encoding  
class FeatureModes(IntEnum):
    FULL = auto()
    EVENT = auto()
    FEATURE = auto()
    TIME = auto()
    ENCODER_DECODER = auto()


class DatasetModes(IntEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class GeneratorModes(IntEnum):
    TOKEN = auto()
    VECTOR = auto()
    HYBRID = auto()


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

    
class MutationMode(IntEnum):
    DELETE = auto()
    INSERT = auto()
    CHANGE = auto()
    SWAP = auto()
    # TODO: Define default assignment for everything none fitting instead of hard coded NONE for 4
    NONE = auto()
    
class InputModeType(Enum):
    TOKEN_INPUT = auto()
    DUAL_INPUT = auto()
    VECTOR_INPUT = auto()

    @staticmethod
    def type(t: FeatureModes):
        if t in [FeatureModes.EVENT]:
            return InputModeType.TOKEN_INPUT
        if t in [FeatureModes.FULL, FeatureModes.TIME]:
            return InputModeType.DUAL_INPUT
        if t in [FeatureModes.FEATURE]:
            return InputModeType.VECTOR_INPUT
        return None