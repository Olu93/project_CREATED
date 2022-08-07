from thesis_commons.modes import TaskModes
from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader100, OutcomeBPIC12Reader75, OutcomeMockReader, OutcomeBPIC12Reader25, OutcomeBPIC12Reader50, OutcomeBPIC12ReaderFull, OutcomeDice4ELReader, OutcomeSepsisReader100, OutcomeSepsisReader25, OutcomeSepsisReader50, OutcomeSepsisReader75, OutcomeTrafficFineReader

DEBUG_VERBOSE = False
DEBUG_INIT_META = True
DEBUG_SKIP_VIZ = False
DEBUG_SKIP_STATS = False

if __name__ == '__main__':
    # TODO: Put debug stuff into configs
    task_mode = TaskModes.OUTCOME_EXTENSIVE_DEPRECATED
    memory_save_mode = TaskModes.OUTCOME_PREDEFINED
    save_preprocessed = True
    # reader = OutcomeMockReader(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeSepsisReader25(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeSepsisReader50(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeSepsisReader75(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeSepsisReader100(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeTrafficFineReader(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader25(debug=DEBUG_VERBOSE, mode=task_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader50(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader75(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader100(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12ReaderFull(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeDice4ELReader(debug=DEBUG_VERBOSE, mode=memory_save_mode).init_log(save_preprocessed).init_meta(DEBUG_INIT_META)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
