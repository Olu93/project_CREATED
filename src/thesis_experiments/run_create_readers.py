from thesis_commons.config import DEBUG_SKIP_VIZ
from thesis_readers import *


if __name__ == '__main__':
    save_preprocessed = True
    reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(DEBUG_SKIP_VIZ)
    reader = OutcomeBPIC12ReaderShort(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(DEBUG_SKIP_VIZ)
    reader = OutcomeBPIC12ReaderMedium(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(DEBUG_SKIP_VIZ)
    reader = OutcomeBPIC12ReaderFull(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(DEBUG_SKIP_VIZ)
