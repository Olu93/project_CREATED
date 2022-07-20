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
from jupyter_constants import map_mrates, map_parts, map_operators, map_operator_shortnames, map_viability, map_erate
# %%
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_params_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%

df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df[df['experiment_name'] == "evolutionary_params"]
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
renamings = {**map_parts, **map_viability, **map_operators, **map_mrates, **map_erate}
df_split = df_configs.copy()
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2, 7], axis=1)
configurations_full_name = configurations.copy().replace(map_operator_shortnames)
df_split = df_split.join(configurations).rename(columns=renamings)
df_split['Model'] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
df_split['num'] = df_split[9].str.replace("num", "").str.replace(".csv", "").astype(int)
# %% plot
topk = 10
df_grouped = df_split.groupby(["num", "Model", "cycle"]).mean().reset_index()
# %% plot
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped, x=x_of_interest, y="viability", ax=faxes, hue='num')
# %%
# %% plot

# %%
last_cycle = df_grouped["cycle"] == df_grouped["cycle"].max()
df_ranked = df_grouped.loc[last_cycle].sort_values(["viability"])[["Model", "num", "viability"]]
df_ranked["rank"] = df_ranked["viability"].rank(ascending=False).astype(int)
df_ranked = df_ranked.rename(columns={"viability": "mean_viability"})
df_ranked
# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on="num", right_on="num").sort_values(["rank"])
df_grouped_ranked
# %%
best_indices = df_grouped_ranked["rank"] <= topk
worst_indices = df_grouped_ranked["rank"] > df_grouped_ranked["rank"].max() - topk
df_grouped_ranked["position"] = "N/A"
df_grouped_ranked.loc[best_indices, "position"] = f"top{topk}"
df_grouped_ranked.loc[worst_indices, "position"] = f"bottom{topk}"
edge_indices = df_grouped_ranked.position != "N/A"
# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked[edge_indices], x=x_of_interest, y="viability", ax=faxes, hue='num')
axes.set_xlabel("Evolution Cycle")
axes.set_ylabel("Mean Viability of the current Population")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked, x="delete-rate", y="viability", ax=faxes)
