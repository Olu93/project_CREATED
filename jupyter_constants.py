import pathlib


ROOT = pathlib.Path('.')
PATH_PAPER = ROOT / "latex" / 'thesis_phase_2' / "generated"
PATH_PAPER_FIGURES = PATH_PAPER / "figures"
PATH_PAPER_TABLES = PATH_PAPER / "tables"

map_parts = {
    "iteration.mean_sparcity": "sparcity",
    "iteration.mean_similarity": "similarity",
    "iteration.mean_feasibility": "feasibility",
    "iteration.mean_delta": "delta",
}

map_erate = {
    'row.operators.mutator.edit_rate':'edit-rate', 
}

map_mrates = {
    'row.operators.mutator.p_delete':'delete-rate',
    'row.operators.mutator.p_insert':'insert-rate',
    'row.operators.mutator.p_change':'change-rate',
    'row.operators.mutator.p_transp':'transp-rate',
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
