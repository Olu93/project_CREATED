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
from jupyter_constants import map_mrates, map_parts, map_operators, map_operator_shortnames, map_viability, map_erate, save_figure
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
cols_operators = list(map_operators.values())[:3]+list(map_operators.values())[4:]
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
x_of_interest = "cycle"
C_MEAN_EVENT_CNT = "Fraction of Events"
C_DELETE = 'DELETE-Rate'
C_INSERT = 'INSERT-Rate'
C_CHANGE = 'CHANGE-Rate'


# %%
renamings = {**map_parts, **map_viability, **map_operators, **map_mrates, **map_erate}
df_split = df_configs.copy()
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2], axis=1)
configurations_full_name = configurations.copy().replace(map_operator_shortnames)
df_split = df_split.join(configurations).rename(columns=renamings)
df_split['Model'] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
df_split['num'] = df_split[9].str.replace("num", "").str.replace(".csv", "").astype(int)
bins = np.linspace(0, 1, 11, endpoint=True)
df_split[C_DELETE] = pd.cut(df_split["delete-rate"], bins=bins)
df_split[C_INSERT] = pd.cut(df_split["insert-rate"], bins=bins)
df_split[C_CHANGE] = pd.cut(df_split["change-rate"], bins=bins)
df_split[C_MEAN_EVENT_CNT] = 1 - df_split["iteration.mean_num_zeros"]
last_cycle = df_split["cycle"] == df_split["cycle"].max()
# %% plot
topk = 10
df_grouped = df_split.groupby(["num", "Model", "cycle"]).mean().reset_index()
last_cycle_grouped = df_grouped["cycle"] == df_grouped["cycle"].max()
# %% plot
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped, x=x_of_interest, y="viability", ax=faxes, hue='num')
# %%
# %% plot
# https://stackoverflow.com/a/43439132/4162265
def plot_mutation_rates(df, x_label='cycle', y_label='viability'):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    faxes = axes.flatten()
    ax = sns.lineplot(data=df, x=x_label, y=y_label, ax=faxes[0], hue=C_DELETE, ci=None, legend=False)
    ax.set_xlabel(f"{C_DELETE}: {x_label}")
    ax = sns.lineplot(data=df, x=x_label, y=y_label, ax=faxes[1], hue=C_INSERT, ci=None, legend=False)
    ax.set_xlabel(f"{C_INSERT}: {x_label}")
    ax = sns.lineplot(data=df, x=x_label, y=y_label, ax=faxes[2], hue=C_CHANGE, ci=None)
    ax.set_xlabel(f"{C_CHANGE}: {x_label}")
    # for ax in axes:
    #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    faxes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"The Effect of Mutation Rates on {y_label.title()}")
    fig.tight_layout()
    return fig, axes

# Notes
# Observations:
# Fraction of Events Declines for any rate. 
# For d-rate under 0.2 this decline stabilizes.
# For i-rate above 0.6 and above this decline is put off but not prevented
# For c-rate between 0.6 and 0.8 this decline is put off but not prevented
# Viability is largely unaffected by any rate
# Feasibility levels always increase
# But they are stuck after hitting a local optimum for all rates 
# Similarity benefits from low change and low-moderate delete and insert 
# Sparcity halts most of the cases.
# Only progresses for low change rates
# Delta halts for all at 1 but is slower for high delete rates and faster for low delete rates
# High change and insert rates lead to faster delta
# Instead of FI use BB

_ = plot_mutation_rates(df_split, 'cycle', C_MEAN_EVENT_CNT)
save_figure("exp2_event_count")
plt.show()
_ = plot_mutation_rates(df_split, 'cycle', y_of_interest)
save_figure("exp2_viability")
plt.show()
_ = plot_mutation_rates(df_split, 'cycle', 'feasibility')
save_figure("exp2_feasibility")
plt.show()
_ = plot_mutation_rates(df_split, 'cycle', 'sparcity')
save_figure("exp2_sparcity")
plt.show()
_ = plot_mutation_rates(df_split, 'cycle', 'similarity')
save_figure("exp2_similarity")
plt.show()
_ = plot_mutation_rates(df_split, 'cycle', 'delta')
save_figure("exp2_delta")
plt.show()
# %%
# %%
df_ranked = df_grouped.loc[last_cycle_grouped].sort_values(["viability"])[["Model", "num", "viability"]]
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
# Steady optimization all diversity is gone
df[['cycle','row.n_population', 'row.n_selection', 'row.n_offspring',
       'row.n_mutated']].groupby("cycle").mean()
