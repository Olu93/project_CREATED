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
# %%
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_configs_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%
map_parts = {
    "iteration.mean_sparcity": "sparcity",
    "iteration.mean_similarity": "similarity",
    "iteration.mean_feasibility": "feasibility",
    "iteration.mean_delta": "delta",
}

map_viability = {"iteration.mean_viability": "viability"}

map_operators = {3: "initiator", 4: "selector", 5: "crosser", 6: "mutator"}

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

df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df[df['experiment_name'] == "evolutionary_configs"]
df_configs
cols_operators = list(map_operators.values())[:-1]
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
x_of_interest = "cycle"
# %%
df_split = df_configs
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2, 7], axis=1)
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
df_no_fi = df_split  #.groupby(["model", "cycle"]).mean().reset_index()
x_of_interest = "cycle"



for row in cols_parts:
    fig, axes = plt.subplots(1, len(cols_operators), figsize=(14, 5), sharey=True)
    faxes = axes.flatten()
    for col, cax in zip(cols_operators, axes):
        df_agg = df_no_fi.groupby([row, col, x_of_interest]).mean().reset_index()  #.replace()
        
        sns.lineplot(data=df_agg, x=x_of_interest, y=row, hue=col, ax=cax, ci=None)
        # ax.invert_xaxis()
        cax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
        cax.set_ylim(0,1)
    fig.tight_layout()
    plt.show()
# %%
# df_no_fi = df_no_fi[df_no_fi['initiator'] != 'FactualInitiator']

# g = sns.FacetGrid(df_melt, row=C_MEAUSURE_TYPE, col=C_OPERATOR_TYPE, hue=C_OPERATOR)
# g.map(sns.lineplot, x_of_interest, C_MEASURE)

# for col, ax in zip(cols_operators, faxes):
#     df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
#     df_agg = df_agg.rename(columns={col: col.upper()})
#     sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
#     # ax.invert_xaxis()
#     ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")

# %% plot
topk = 10
df_grouped = df_split.groupby(cols_operators + ["Model","cycle"]).mean().reset_index()

# %%
last_cycle = df_grouped["cycle"] == df_grouped["cycle"].max()
df_ranked = df_grouped.loc[last_cycle].sort_values(["viability"])[["Model", "viability"]]
df_ranked["rank"] = df_ranked["viability"].rank(ascending=False).astype(int)
df_ranked = df_ranked.rename(columns={"viability":"mean_viability"})
df_ranked
# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on="Model", right_on="Model").sort_values(["rank"])
df_grouped_ranked
# %%
best_indices = df_grouped_ranked["rank"] <= topk
worst_indices = df_grouped_ranked["rank"] > df_grouped_ranked["rank"].max()-topk
df_grouped_ranked["position"] = "N/A"
df_grouped_ranked.loc[best_indices,"position"] = f"top{topk}"
df_grouped_ranked.loc[worst_indices,"position"] = f"bottom{topk}"
edge_indices = df_grouped_ranked.position != "N/A"
# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked[edge_indices], x=x_of_interest, y="viability", ax=faxes, hue='Model')
axes.set_xlabel("Evolution Cycle")
axes.set_ylabel("Mean Viability of the current Population")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# fig.tight_layout()
# g = sns.relplot(
#     data=df_grouped[df_grouped.position != "N/A"],
#     x=x_of_interest,
#     y=y_of_interest,
#     hue="Model",
#     kind="line",
#     row="position",
#     # col_wrap=3,
#     aspect=2.5,
# )
# %%
df_table = df_grouped[df_grouped.position != "N/A"][last_cycle][[y_of_interest, "position"] + cols_parts]
df_table
# %%
x_of_interest = "cycle"
df_melted = pd.melt(df_important, id_vars=set(df_important.columns) - set(map_parts), value_vars=map_parts)
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
    col="variable",
    col_wrap=2,
    aspect=1.2,
)
# sns.lineplot(data=df_important, x=x_of_interest, y="iteration.mean_similarity", ax=faxes[1], hue='Model')
# sns.lineplot(data=df_important, x=x_of_interest, y="iteration.mean_sparcity", ax=faxes[3], hue='Model')
# sns.lineplot(data=df_important, x=x_of_interest, y="iteration.mean_feasibility", ax=faxes[0], hue='Model')
# sns.lineplot(data=df_important, x=x_of_interest, y="iteration.mean_delta", ax=faxes[2], hue='Model')
# %%
fig, ax = plt.subplots(figsize=(12, 12))
sns.lineplot(data=df_important, x="cycle", y="iteration.mean_feasibility", hue="Model")
# %%