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

# %%
PATH = pathlib.Path('results/models_specific/specific_model_results.csv')
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
cols_operator = ['initiator', 'selector', 'crosser', 'mutator', 'recombiner']
# %%
df_grouped = df_configs
configurations = df_grouped["model"].str.split("_", expand=True).drop([0, 1, 2, 8], axis=1).rename(columns={
    3: "initiator",
    4: "selector",
    5: "crosser",
    6: "mutator",
    7: "recombiner"
})
df_split = df_grouped.join(configurations)
# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df_split  #.groupby(["model", "cycle"]).mean().reset_index()
y_of_interest = "iteration.mean_viability"
x_of_interest = "cycle"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes):
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    # ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df_split
y_of_interest = "iteration.mean_feasibility"
x_of_interest = "cycle"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes):
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    # ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df_split
y_of_interest = "iteration.mean_sparcity"
x_of_interest = "cycle"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes):
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    # ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df_split
y_of_interest = "iteration.mean_similarity"
x_of_interest = "cycle"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes):
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    # ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df_split
y_of_interest = "iteration.mean_delta"
x_of_interest = "cycle"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes):
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()  #.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    # ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")
# %% plot
topk = 10
df_grouped = df_split.groupby(cols_operator + ["cycle"]).mean().reset_index()
df_grouped['Model'] = df_grouped[cols_operator].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# %%
last_cycle = df_grouped["cycle"] == df_grouped["cycle"].max()
df_top_performers = df_grouped.loc[last_cycle].sort_values(["iteration.mean_viability"])[["Model", "iteration.mean_viability"]]
df_top_performers
# %%
least_performers = df_top_performers.head(topk)
best_performers = df_top_performers.tail(topk)

all_important = pd.concat([least_performers, best_performers])
all_important

# %%
df_important = df_grouped[df_grouped["Model"].isin(all_important["Model"].values)]
df_important
# %%
fig, axes = plt.subplots(1, 1, figsize=(15, 10), sharey=True)
faxes = axes  #.flatten()
y_of_interest = "iteration.mean_viability"
x_of_interest = "cycle"
sns.lineplot(data=df_important, x=x_of_interest, y=y_of_interest, ax=faxes, hue='Model')
# %%
x_of_interest = "cycle"
col_components = ["iteration.mean_similarity", "iteration.mean_sparcity", "iteration.mean_feasibility", "iteration.mean_delta"]
df_melted = pd.melt(df_important, id_vars=set(df_important.columns) - set(col_components), value_vars=col_components)

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

# %%
