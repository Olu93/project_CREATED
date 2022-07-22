import pathlib
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path('.')
PATH_PAPER = ROOT / "latex" / 'thesis_phase_2'
PATH_PAPER_FIGURES = PATH_PAPER / "figures/generated"
PATH_PAPER_TABLES = PATH_PAPER / "tables/generated"

C_DELETE = 'Delete-Rate'
C_INSERT = 'Insert-Rate'
C_CHANGE = 'Change-Rate'
C_RANGE_DELETE = 'Binned Delete-Rate'
C_RANGE_INSERT = 'Binned Insert-Rate'
C_RANGE_CHANGE = 'Binned Change-Rate'
C_SPARCITY = 'Sparcity'
C_SIMILARITY = 'Similarity'
C_FEASIBILITY = 'Feasibility'
C_DELTA = 'Delta'
C_VIABILITY = "Viability"
C_XLABEL_CYCLES = "Evolution Cycles"
C_MODEL_CONFIG = "Model Configuration"
C_VIABILITY_COMPONENT = "Viability-Component"
C_MRATE = "Mutation Rate"
COLS_MRATES = [C_DELETE, C_INSERT, C_CHANGE]
COLS_MRATES_CLS = ["Binned " + tmp for tmp in [C_DELETE, C_INSERT, C_CHANGE]]
COLS_VIAB_COMPONENTS = [C_SIMILARITY, C_SPARCITY, C_FEASIBILITY, C_DELTA]

map_parts = {
    "iteration.mean_sparcity": C_SPARCITY,
    "iteration.mean_similarity": C_SIMILARITY,
    "iteration.mean_feasibility": C_FEASIBILITY,
    "iteration.mean_delta": C_DELTA,
}

map_erate = {
    'row.operators.mutator.edit_rate': 'edit-rate',
}

map_mrates = {
    'row.operators.mutator.p_delete': C_DELETE,
    'row.operators.mutator.p_insert': C_INSERT,
    'row.operators.mutator.p_change': C_CHANGE,
    # 'row.operators.mutator.p_transp': 'transp-rate',
    # 'row.operators.mutator.p_none':'none',
}

map_viability = {"iteration.mean_viability": "viability"}

map_operators = {3: "initiator", 4: "selector", 5: "crosser", 6: "mutator", 7: "recombiner"}

map_operator_shortnames = {
    "CBI": "CaseBasedInitiator",
    "DBI": "DistributionBasedInitiator",
    "DI": "RandomInitiator",
    "FI": "FactualInitiator",
    "ES": "ElitismSelector",
    "TS": "RandomWheelSelector",
    "RWS": "TournamentSelector",
    "OPC": "OnePointCrosser",
    "TPC": "TwoPointCrosser",
    "UC": "UniformCrosser",
    "DDM": "DistributionBasedMutator",
    "DM": "RandomMutator",
    "BBR": "BestBreedMerger",
    "FIR": "FittestPopulationMerger",
}


def save_figure(title: str):
    plt.savefig((PATH_PAPER_FIGURES / title).absolute(), bbox_inches="tight")


def save_table(table: Union[str, pd.DataFrame], filename: str):
    if isinstance(table, pd.DataFrame):
        table = table.style.format(escape="latex").to_latex()
    destination = PATH_PAPER_TABLES / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table)
