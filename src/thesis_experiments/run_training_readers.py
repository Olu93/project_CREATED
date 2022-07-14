from thesis_commons.modes import TaskModes
from thesis_readers import OutcomeMockReader, OutcomeBPIC12ReaderShort, OutcomeBPIC12ReaderMedium, OutcomeBPIC12ReaderFull

DEBUG_SKIP_VIZ = True
DEBUG_SKIP_STATS = False

if __name__ == '__main__':
    # TODO: Put debug stuff into configs
    save_preprocessed = True
    reader = OutcomeMockReader(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ,skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeBPIC12ReaderShort(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ,skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeBPIC12ReaderMedium(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ,skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeBPIC12ReaderFull(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ,skip_stats=DEBUG_SKIP_STATS)