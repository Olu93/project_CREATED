from thesis_commons.modes import TaskModes
from thesis_readers import OutcomeMockReader, OutcomeBPIC12ReaderShort, OutcomeBPIC12ReaderMedium, OutcomeBPIC12ReaderFull

if __name__ == '__main__':
    # TODO: Put debug stuff into configs
    save_preprocessed = True
    reader = OutcomeMockReader(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderShort(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderMedium(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)
    reader = OutcomeBPIC12ReaderFull(debug=True, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(True)