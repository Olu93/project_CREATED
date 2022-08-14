import pathlib
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path('.')
PATH_PAPER = ROOT / "latex" / 'thesis_phase_2'
PATH_PAPER_FIGURES = PATH_PAPER / "figures/generated"
PATH_PAPER_TABLES = PATH_PAPER / "tables/generated"

C_INITIATOR = 'Initiator'
C_SELECTOR = 'Selector'
C_CROSSER = 'Crosser'
C_MUTATOR = 'Mutator'
C_RECOMBINER = 'Recombiner'
COLS_OPERATORS = [C_INITIATOR, C_SELECTOR, C_CROSSER, C_MUTATOR, C_RECOMBINER]

C_DELETE = 'Delete-Rate'
C_INSERT = 'Insert-Rate'
C_CHANGE = 'Change-Rate'
C_TRANSP = 'Transpose-Rate'
C_RANGE_DELETE = 'Binned Delete-Rate'
C_RANGE_INSERT = 'Binned Insert-Rate'
C_RANGE_CHANGE = 'Binned Change-Rate'
COLS_MRATES = [C_DELETE, C_INSERT, C_CHANGE]
COLS_MRATES_CLS = ["Binned " + tmp for tmp in [C_DELETE, C_INSERT, C_CHANGE]]

C_SPARCITY = 'Sparcity'
C_SIMILARITY = 'Similarity'
C_FEASIBILITY = 'Feasibility'
C_DELTA = 'Delta'
C_VIABILITY = "Viability"
COLS_VIAB_COMPONENTS = [C_SIMILARITY, C_SPARCITY, C_FEASIBILITY, C_DELTA]
C_MEAN_VIABILITY = "Mean Viability"
C_FACTUAL_OUTCOME = "Factual Outcome"
C_TARGET_OUTCOME = "Target Outcome"
C_PRED_OUTCOME = "Predicted Outcome"
C_RUN_MASK = "Used Measure Components"
C_RUN_NO = "Num Simulations"
C_EXPERIMENT_ID = "Num Experiments"
C_CYCLE = "Iterative Cycle"
C_CYCLE_TERMINATION = "Termination Point"
# C_CYCLE_RELATIVE = C_CYCLE + " (scaled between 0.0 & 1.0)"
C_CYCLE_NORMED = "Iterative Cycles (normalized)"
C_MODEL_CONFIG = "Model Configuration"
C_VIABILITY_COMPONENT = "Viability-Component"
C_MRATE = "Mutation Rate"
C_ITERATION = 'Iteration'
C_INSTANCE = 'Instance'
C_MODEL = "Model"
C_RANK = "Rank"
C_OPACITY = "Opacity"
C_POSITION = "Position"
C_PAD_RATIO = "Padding Ratio"
C_EVT_RATIO = "Event Ratio"
C_SHORT_NAME = "Model (Abbr. Name)"
C_FULL_NAME = "Full Name"
C_MODEL_NAME = "Generator Type"
C_EXPERIMENT_NAME = "Experiment"
C_SIMULATION_NAME = "Simulation"
C_ID = "Identifier"
C_DURATION = "Processing Time (sec.)"
C_PRED_SCORE = "Prediction Score"
C_MAX_LEN = "Max. Seq. Length"
C_NUM_ZEROS = "Num. Paddings"
C_NUM_EVENTS = "Num. Events"
map_parts = {
    "iteration.mean_sparcity": C_SPARCITY,
    "iteration.mean_similarity": C_SIMILARITY,
    "iteration.mean_feasibility": C_FEASIBILITY,
    "iteration.mean_delta": C_DELTA,
}

map_parts_overall = {
    "sparcity": C_SPARCITY,
    "similarity": C_SIMILARITY,
    "feasibility": C_FEASIBILITY,
    "delta": C_DELTA,
}

map_erate = {
    'row.operators.mutator.edit_rate': 'edit-rate',
}

map_mrates = {
    'row.operators.mutator.p_delete': C_DELETE,
    'row.operators.mutator.p_insert': C_INSERT,
    'row.operators.mutator.p_change': C_CHANGE,
    'row.operators.mutator.p_transp': C_TRANSP,
    # 'row.operators.mutator.p_none':'none',
}

map_viability_specifics = {"iteration.mean_viability": C_VIABILITY}
map_viability_overall = {"viability": C_VIABILITY}

map_operators = {2: "initiator", 3: "selector", 4: "crosser", 5: "mutator", 6: "recombiner"}


map_operator_long2short = {
    'CaseBasedInitiator': 'CBI',
    'DistributionBasedInitiator': 'DBI',
    'RandomInitiator': 'DI',
    'FactualInitiator': 'FI',
    'ElitismSelector': 'ES',
    'RandomWheelSelector': 'TS',
    'TournamentSelector': 'RWS',
    'OnePointCrosser': 'OPC',
    'TwoPointCrosser': 'TPC',
    'UniformCrosser': 'UC',
    'DistributionBasedMutator': 'DDM',
    'RandomMutator': 'DM',
    'BestBreedMerger': 'BBR',
    'FittestPopulationMerger': 'FPM',
    'SamplingBasedInitiator': 'SBI',
    'SamplingBasedMutator': 'SBM',
    'FittestSurvivorRecombiner': 'FSR',
}

map_operator_short2long = {v: k for k, v in map_operator_long2short.items()}

map_operator_specifics = {
    "row.operators.initiator.type": C_INITIATOR,
    "row.operators.selector.type": C_SELECTOR,
    "row.operators.crosser.type": C_CROSSER,
    "row.operators.mutator.type": C_MUTATOR,
    "row.operators.recombiner.type": C_RECOMBINER,
}

map_ds_names = {
    "OutcomeDice4ELReader": "DiCE4EL",
    "OutcomeBPIC12Reader25": "BPIC12-25",
    "OutcomeBPIC12Reader50": "BPIC12-50",
    "OutcomeBPIC12Reader75": "BPIC12-75",
    "OutcomeBPIC12Reader100": "BPIC12-100",
    # "OutcomeBPIC12ReaderFull": "BPIC12-Full",
    "OutcomeSepsisReader25": "Sepsis-25",
    "OutcomeSepsisReader50": "Sepsis-50",
    "OutcomeSepsisReader75": "Sepsis-75",
    "OutcomeSepsisReader100": "Sepsis-100",
    "OutcomeTrafficFineReader": "TrafficFines",
}

# ------- Current Datasets/Overall ---------
# 1 Run -> 1 Model Id: N Factuals == N Instances
# 1 Instance -> 1 Factual Group Id: Multiple Instances == Cycles
# 1 Iteration -> 1 Cycle Id: 
# Row -> Counterfactual Id


# -------------------------
# Hierarchy:
# - Run
# - - Multiple Instances        

# - Instance
# - - Multiple Iterations 
# - - A models' statistic over all instances 

# - Iteration
# - - Multiple Rows
# - - A models' statistic for one case  

# - Row/Cycle/Evo-Iteration
# - - Case Level Data
# - - A models' row data

# - Outcome overall results and Evo Cycle speecifics are not conforming 


map_name_specifics = {
    'instance.short_name':C_SHORT_NAME,
    'instance.full_name':C_FULL_NAME,
}

map_name_overall = {
    'run.short_name':C_SHORT_NAME,
    'run.full_name':C_FULL_NAME,
    'run.mask':C_RUN_MASK,
    'run.no':C_RUN_NO,
    'experiment':C_EXPERIMENT_ID,
    'result_outcome':C_PRED_OUTCOME,
    'source_outcome':C_FACTUAL_OUTCOME,
    'target_outcome':C_TARGET_OUTCOME,
    "likelihood": C_PRED_SCORE,
    "cf_num_zeros": C_NUM_ZEROS,
    "wrapper.max_len": C_MAX_LEN,
    
}

map_specifics = {
    "row.num_cycle": C_CYCLE,
    "instance.no": C_INSTANCE,
    "iteration.no": C_ITERATION,
    "filename": C_SIMULATION_NAME,
    "wrapper_type": C_MODEL_NAME,
    "experiment_name": C_EXPERIMENT_NAME,
    "iteration.mean_num_zeros":C_PAD_RATIO,
    "instance.duration_sec":C_DURATION,
    **map_name_specifics,
    **map_parts,
    **map_viability_specifics,
    **map_mrates,
    **map_erate,
    **map_operator_specifics,
}
map_overall = {
    "row.num_cycle": C_CYCLE,
    "run.short_name": C_SHORT_NAME,
    "instance.no": C_INSTANCE,
    "run.duration_sec": C_DURATION,
    "iteration.no": C_ITERATION,
    "filename": C_SIMULATION_NAME,
    "experiment_name": C_EXPERIMENT_NAME,
    "rank": C_RANK,
    **map_name_overall,
    **map_parts,
    **map_viability_overall,
    **map_mrates,
    **map_erate,
    **map_parts_overall,
    # **map_operator_specifics,
}

TBL_FORMAT_RULES = {
   ("Numeric", "Integers"): '\${}',
   ("Numeric", "Floats"): '{:.6f}',
   ("Non-Numeric", "Strings"): str.upper
}


def save_figure(title: str):
    plt.savefig((PATH_PAPER_FIGURES / title).absolute(), bbox_inches="tight")


def save_table(table: Union[str, pd.DataFrame], filename: str):
    if isinstance(table, pd.DataFrame):
        table = table.style.format(escape="latex").to_latex()
    destination = PATH_PAPER_TABLES / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table.replace("_", "-"))

def remove_name_artifacts(series: pd.Series):
    result = series
    result = result.str.replace("ES_EGW_", "")
    result = result.str.replace("_IM", "")
    result = result.str.replace(".csv", "")
    result = result.str.replace("_", "-")
    return result