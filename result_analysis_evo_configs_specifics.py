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
from jupyter_constants import PATH_PAPER_FIGURES, map_mrates, map_parts, map_operators, map_operator_shortnames, map_viability, map_erate, save_figure, save_table

# %%
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_configs_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%


df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df[df['experiment_name'] == "evolutionary_configs"]
df_configs
cols_operators = list(map_operators.values())[:3] + list(map_operators.values())[4:]
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
x_of_interest = "cycle"
# %%
df_split = df_configs
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2], axis=1)
configurations_full_name = configurations.copy().replace(map_operator_shortnames)
df_split = df_split.join(configurations).rename(columns={**map_parts, **map_viability, **map_operators})
df_split['Model'] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# # %%
# df_melt = df_split.groupby(cols_operators + ["cycle"]).mean().reset_index().copy()
# df_melt = df_melt.melt(id_vars=cols_operators + [y_of_interest, x_of_interest], value_vars=cols_parts, var_name=C_MEAUSURE_TYPE,
#                        value_name=C_MEASURE)  #.groupby(["model", "cycle"]).mean().reset_index()
# df_melt = df_melt.melt(id_vars=[y_of_interest, x_of_interest, C_MEAUSURE_TYPE, C_MEASURE], value_vars=cols_operators[:-2], var_name=C_OPERATOR_TYPE,
#                        value_name=C_OPERATOR)  #.groupby(["model", "cycle"]).mean().reset_index()
# num_otypes = df_melt[C_OPERATOR_TYPE].nunique()
# num_mtypes = df_melt[C_MEAUSURE_TYPE].nunique()

# fig, axes = plt.subplots(num_mtypes, num_otypes, figsize=(12, 8), sharey=True)
# # faxes = axes.flatten()

# for (row_idx, row_df), row_axes in zip(df_melt.groupby([C_MEAUSURE_TYPE]), axes):
#     for (col_idx, col_df), caxes in zip(row_df.groupby([C_OPERATOR_TYPE]), row_axes):
#         sns.lineplot(data=col_df, x=x_of_interest, y=C_MEASURE, hue=C_OPERATOR, ax=caxes, legend=None)


# %%
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
# df_no_fi = df_split[df_split['initiator'] != 'FactualInitiator']
df_tmp = df_split.copy()  #.groupby(["model", "cycle"]).mean().reset_index()
x_of_interest = "cycle"


for row in cols_parts:
    fig, axes = plt.subplots(1, len(cols_operators), figsize=(12, 3), sharey=True)
    faxes = axes.flatten()
    for col, cax in zip(cols_operators, axes):
        df_agg = df_tmp.groupby([row, col, x_of_interest]).median().reset_index()  #.replace()
        
        sns.lineplot(data=df_agg, x=x_of_interest, y=row, hue=col, ax=cax, ci=None)
        # ax.invert_xaxis()
        cax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
        cax.set_ylim(0,1)
    fig.tight_layout()
    save_figure(f"exp1_{row}")
    plt.show()
# NOTE: 
# The initiator governs the starting point in terms of sparcity and similarity after which similarity hardly changes
# Feasibility increases only if DBI is used.
# Feasibility increases fastest for ES 
# Feasibility is slightly higher for OPC instead of TPC
# Feasibility is lower for BBR but FSR reaches plateau
# Delta is reached fairly quickly

# %%
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
# df_split["pareto_efficient"] = False
# subset = df_split["cycle"]==df_split["cycle"].max()
# subset = df_split["cycle"] > df_split["cycle"].min()
# df_split.loc[subset, "pareto_efficient"] = is_pareto_efficient(df_split.loc[subset, cols_parts[:2]].values)
# sns.lineplot(data=df_split[df_split["pareto_efficient"]==True], y="sparcity", x="similarity")
# df_split["pareto_efficient"] = False

df_split["pareto_efficient"] = False
subset = df_split["cycle"]==df_split["cycle"].max()
subset = df_split["cycle"] != None
df_split.loc[subset, "pareto_efficient"] = is_pareto_efficient(df_split.loc[subset, ["feasibility", "similarity"]].values)
sns.lineplot(data=df_split[df_split["pareto_efficient"]==True], y="feasibility", x="similarity")

# %% plot
topk = 10

df_grouped = df_split.groupby(cols_operators + ["Model","cycle"]).median().reset_index() # Grp. over instances

# %%
last_cycle = df_grouped["cycle"] == df_grouped["cycle"].max()
df_ranked = df_grouped.loc[last_cycle].sort_values(["viability"])[["Model", "viability"] + cols_parts]
df_ranked["rank"] = df_ranked["viability"].rank(ascending=False).astype(int)
df_ranked = df_ranked.rename(columns={"viability":"mean_viability"})
tmp_len = len(df_ranked["rank"])
df_table_edge_cases = df_ranked[~df_ranked["rank"].between(1+topk, tmp_len-topk)].set_index("rank")
display(df_table_edge_cases)

save_table(df_table_edge_cases, "exp1_edge_cases")

# NOTE: 
# Worst combination is RI with BBR, because it starts bad and keeps bad individuals in population.
# Top combination is DBI with TPC as 4 out of 5 in top5 are DBI.
# ES seems to work best outright.
# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on="Model", right_on="Model").sort_values(["rank"])
df_grouped_ranked
# %%
best_indices = df_grouped_ranked["rank"] <= topk
worst_indices = df_grouped_ranked["rank"] > df_grouped_ranked["rank"].max()-topk
df_grouped_ranked["position"] = "N/A"
df_grouped_ranked.loc[best_indices,"position"] = f"top{topk}"
df_grouped_ranked.loc[worst_indices,"position"] = f"bottom{topk}"
edge_indices = df_grouped_ranked["position"] != "N/A"
df_grouped_ranked
# %%
# pd.concat([])

# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked[edge_indices], x=x_of_interest, y="viability", ax=faxes, hue='Model', estimator="median")
axes.set_xlabel("Evolution Cycles")
axes.set_ylabel("Mean Viability of the current Population")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
save_figure("exp1_effect_on_viability_top10_last10")

# NOTE: 
# Requires 50 cycles at least to be conclusive

# %%
df_table = df_grouped[last_cycle][[y_of_interest] + cols_parts]
df_table
# %%
x_of_interest = "cycle"
df_melted = pd.melt(df_grouped, id_vars=set(df_grouped.columns) - set(cols_parts), value_vars=cols_parts, var_name="Measure")
# y_of_interest = "iteration.mean_viability"
df_melted
# %%
# fig, axes = plt.subplots(figsize=(15,15), sharey=True)
# faxes = axes.flatten()
g = sns.relplot(
    data=df_melted,
    x="cycle",
    y="value",
    hue="Model",
    kind="line",
    col="Measure",
    col_wrap=2,
    aspect=1.2,
)
g.set_xlabels("Evolution Cycles")
g.set_ylabels("Value")
# g.fig.suptitle("Effect of Model Configuration over Evolution Cycles")
g.tight_layout()
save_figure("exp1_effect_on_measures")
# NOTE: 
# Gap seems to be coming from sparcity and similarity
# Feasibility reaches higher levels with DBI initiation

# %%
# NOTE: 
# Choose DBI, ES, 


# MAJOR TODO: Investigate pareto viability measure