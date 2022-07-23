from thesis_commons.modes import TaskModes
from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader100, OutcomeBPIC12Reader75, OutcomeMockReader, OutcomeBPIC12Reader25, OutcomeBPIC12Reader50, OutcomeBPIC12ReaderFull, OutcomeDice4ELReader, OutcomeSepsisReader100, OutcomeSepsisReader25, OutcomeSepsisReader50, OutcomeSepsisReader75, OutcomeTrafficFineReader

DEBUG_SKIP_VIZ = True
DEBUG_SKIP_STATS = False

if __name__ == '__main__':
    # TODO: Put debug stuff into configs

    save_preprocessed = True
    # reader = OutcomeMockReader(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeSepsisReader25(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeSepsisReader50(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeSepsisReader75(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeSepsisReader100(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeTrafficFineReader(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    reader = OutcomeDice4ELReader(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader25(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader50(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader75(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12Reader100(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
    # reader = OutcomeBPIC12ReaderFull(debug=False, mode=TaskModes.OUTCOME_PREDEFINED).init_log(save_preprocessed).init_meta(False)
    # reader.save(skip_viz=DEBUG_SKIP_VIZ, skip_stats=DEBUG_SKIP_STATS)
