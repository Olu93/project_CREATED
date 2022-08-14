# %%
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from sklearn import metrics
from IPython.display import display
from scipy import stats
from scipy import spatial
from jupyter_constants import *
# %%
df = pd.DataFrame([
    ["Initiation", "Random-Initiation", ""],
    ["Initiation", "Sample-Based-Initiation", ""],
    ["Initiation", "Case-Based-Initiation", ""],
    ["Selection", "Roulette-Wheel-Selection", "Sample-Size"],
    ["Selection", "Tournament-Selection", "Sample-Size"],
    ["Selection", "Elitism-Selection", "Sample-Size"],
    ["Crossing", "Uniform-Crossing", "Crossover-Rate"],
    ["Crossing", "One-Point-Crossing", ""],
    ["Crossing", "Two-Point-Crossing", ""],
    ["Mutation", "Random-Mutation", "Mutation-Rates"],
    ["Mutation", "Sampling-Based-Mutation", "Mutation-Rates"],
    ["Recombination", "Fittest-Survivor-Recombination", "Population-Size"],
    ["Recombination", "Best-of-Breed-Recombination", "Population-Size"],
    ["Recombination", "Ranked-Recombination", "Population-Size"],
], columns=["Operator-Type", "Operator", "Hyperparameter"])

df
# df.pivot(index="Operation-Type", values=["Operators"], columns=)
# %%
df = df.groupby("Operator-Type").apply(lambda x: x.reset_index()).drop(["index", "Operator-Type"], axis=1)

df_table = df#.sort_values(C_VIABILITY, ascending=False)
df_styled = df_table.style.format(
    # escape='latex',
    precision=5,
    na_rep='',
    thousands=" ",
) #.hide(None)
df_latex = df_styled.to_latex(
    multicol_align='l',
    multirow_align='t',
    clines='skip-last;data',
    # column_format='l',
    # caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
    # label=f"tbl:example-cf-{'-'.join(config_name)}",
    hrules=True,
)  #.replace("15 214", "")
save_table(df_latex, "operators-table")

df_styled
# %%
