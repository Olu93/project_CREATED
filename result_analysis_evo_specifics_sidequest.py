# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from sklearn import metrics
from IPython.display import display
from scipy import stats
from scipy import spatial
import itertools as it
from jupyter_constants import *

# %%
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_sidequest_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%


df = original_df.copy()


cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
# %%
df_split = df
df_split = df_split.rename(columns=map_specifics)
df_split[C_SIMULATION_NAME] = df_split[C_SIMULATION_NAME].str.replace("ES_EGW_", "").str.replace("_IM.csv", "")
df_split

# %%
df_split = df_split[df_split[C_INITIATOR]!="FactualInitiator"]

# %%
# NOTE: 
# The initiator governs the starting point in terms of sparcity and similarity after which similarity hardly changes
# Feasibility increases only if DBI is used.
# Feasibility increases fastest for ES 
# Feasibility is slightly higher for OPC instead of TPC
# Feasibility is lower for BBR but FSR reaches plateau
# Delta is reached fairly quickly
df_tmp = df_split.copy() 
x_of_interest = C_CYCLE


for row in COLS_VIAB_COMPONENTS:
    fig, axes = plt.subplots(1, len(COLS_OPERATORS), figsize=(25, 5), sharey=True)
    faxes = axes.flatten()
    for col, cax in zip(COLS_OPERATORS, axes):
        df_agg = df_tmp.groupby([row, col, x_of_interest]).median().reset_index()  #.replace()
        
        sns.lineplot(data=df_agg, x=x_of_interest, y=row, hue=col, ax=cax, ci=None)
        # ax.invert_xaxis()
        cax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
        cax.set_ylim(0,1)
    fig.tight_layout()
    # save_figure(f"exp1_{row}")
    plt.show()


# %% plot
# NOTE: 
# Worst combination is RI with BBR, because it starts bad and keeps bad individuals in population.
# Top combination is DBI with TPC as 4 out of 5 in top5 are DBI.
# ES seems to work best outright.
topk = 10
df_grouped = df_split.groupby(COLS_OPERATORS + [C_SIMULATION_NAME,C_CYCLE]).median().reset_index() # Grp. over instances
last_cycle = df_grouped[C_CYCLE] == df_grouped[C_CYCLE].max()
df_ranked = df_grouped.loc[last_cycle].sort_values([C_VIABILITY])[[C_SIMULATION_NAME, C_VIABILITY] + COLS_VIAB_COMPONENTS]
df_ranked[C_RANK] = df_ranked[C_VIABILITY].rank(ascending=False).astype(int)
df_ranked = df_ranked.rename(columns={C_VIABILITY:"mean_viability"})
tmp_len = len(df_ranked[C_RANK])
df_table_edge_cases = df_ranked[~df_ranked[C_RANK].between(1+topk, tmp_len-topk)].set_index(C_RANK)
display(df_table_edge_cases)

save_table(df_table_edge_cases, "exp1_edge_cases")


# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on=C_SIMULATION_NAME, right_on=C_SIMULATION_NAME).sort_values([C_RANK])
df_grouped_ranked
# %%
best_indices = df_grouped_ranked[C_RANK] <= topk
worst_indices = df_grouped_ranked[C_RANK] > df_grouped_ranked[C_RANK].max()-topk
df_grouped_ranked[C_POSITION] = "N/A"
df_grouped_ranked.loc[best_indices,C_POSITION] = f"top{topk}"
df_grouped_ranked.loc[worst_indices,C_POSITION] = f"bottom{topk}"
df_grouped_ranked[C_OPACITY] = 0
df_grouped_ranked.loc[df_grouped_ranked[C_POSITION]=="N/A",C_OPACITY] = 0.1
df_grouped_ranked.loc[df_grouped_ranked[C_POSITION]!="N/A",C_OPACITY] = 1
edge_indices = df_grouped_ranked[C_POSITION] != "N/A"
df_grouped_ranked


# %%
# NOTE: 
# Requires 50 cycles at least to be conclusive
fig, axes = plt.subplots(1, 1, figsize=(8, 8), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked[edge_indices], x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_SIMULATION_NAME, estimator="median", style=C_POSITION, alpha=1)
sns.lineplot(data=df_grouped_ranked[~edge_indices], x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_SIMULATION_NAME, estimator="median", alpha=0.1, legend=None)
axes.set_xlabel(C_CYCLE)
axes.set_ylabel("Mean Viability of the current Population")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# save_figure("exp1_effect_on_viability_top10_last10")



# %%
df_table = df_grouped[last_cycle][[C_VIABILITY] + COLS_VIAB_COMPONENTS]
df_table
# %%
df_melted = pd.melt(df_grouped, id_vars=set(df_grouped.columns) - set(COLS_VIAB_COMPONENTS), value_vars=COLS_VIAB_COMPONENTS, var_name="Measure")
# y_of_interest = "iteration.mean_viability"
df_melted
# %%
# NOTE: 
# Gap seems to be coming from sparcity and similarity
# Feasibility reaches higher levels with DBI initiation

g = sns.relplot(
    data=df_melted,
    x=C_CYCLE,
    y="value",
    hue=C_SIMULATION_NAME,
    kind="line",
    col="Measure",
    col_wrap=2,
    aspect=1.2,
)
g.set_xlabels(C_CYCLE)
g.set_ylabels("Value")
# g.fig.suptitle("Effect of Model Configuration over Evolution Cycles")
g.tight_layout()
# save_figure("exp1_effect_on_measures")


# %%
# NOTE: 
# Choose DBI, ES, 
