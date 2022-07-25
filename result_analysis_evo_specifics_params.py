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
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_params_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%

df = original_df.copy()


# %%
df_split = df.copy()
df_split = df_split.rename(columns=map_specifics)
df_split[C_MODEL] = df_split[C_SIMULATION_NAME].str.split("_num", expand=True).iloc[:, 0]
df_split[C_SIMULATION_NAME] = df_split[C_SIMULATION_NAME].str.split("_num", expand=True).iloc[:, -1].str.replace(".csv", "").astype(int)
df_split[C_SIMULATION_NAME] = pd.Categorical(df_split[C_SIMULATION_NAME])
df_split[C_EVT_RATIO] = 1-df_split[C_PAD_RATIO]
bins = np.linspace(0, 1, 11, endpoint=True)
df_split[C_RANGE_DELETE] = pd.cut(df_split[C_DELETE], bins=bins)
df_split[C_RANGE_INSERT] = pd.cut(df_split[C_INSERT], bins=bins)
df_split[C_RANGE_CHANGE] = pd.cut(df_split[C_CHANGE], bins=bins)
df_split
# renamings = {**map_parts, **map_viability_specifics, **map_operators, **map_mrates, **map_erate}
# df_split = df_configs.copy()
# configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2], axis=1)
# configurations_full_name = configurations.copy().replace(map_operator_short2long)
# df_split = df_split.join(configurations).rename(columns=renamings)
# df_split['Model'] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
# df_split['exp'] = df_split[9].str.replace("num", "").str.replace(".csv", "").astype(int)
# df_split[C_MEAN_EVENT_CNT] = 1 - df_split["iteration.mean_num_zeros"]
# last_cycle = df_split[C_CYCLE] == df_split[C_CYCLE].max()
# df_split["Mutation Rate"] = "" + df_split[C_DELETE].apply(lambda x: f"D={x:.3f}") + " " + df_split[C_INSERT].apply(
#     lambda x: f"I={x:.3f}") + " " + df_split[C_CHANGE].apply(lambda x: f"C={x:.3f}")
# df_split["id"] = df_split[C_SIMULATION_NAME].astype(str) + "-" + df_split["iteration"].astype(str) + "-" + df_split[C_CYCLE].astype(str)


# %% plot
topk = 5
df_grouped = df_split.groupby([C_SIMULATION_NAME, C_MODEL, C_CYCLE]).mean().reset_index()
last_cycle_grouped = df_grouped[C_CYCLE] == df_grouped[C_CYCLE].max()
# %% plot
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped, x=C_CYCLE, y=C_VIABILITY, ax=faxes, hue=C_SIMULATION_NAME)


# %%
# %% plot
# https://stackoverflow.com/a/43439132/4162265
def plot_mutation_rates(df, x_label=C_CYCLE, y_label=C_VIABILITY):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
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

_ = plot_mutation_rates(df_split, C_CYCLE, C_EVT_RATIO)
save_figure("exp2_event_count")
plt.show()
_ = plot_mutation_rates(df_split, C_CYCLE, C_VIABILITY)
save_figure("exp2_viability")
plt.show()
_ = plot_mutation_rates(df_split, C_CYCLE, C_FEASIBILITY)
save_figure("exp2_feasibility")
plt.show()
_ = plot_mutation_rates(df_split, C_CYCLE, C_SPARCITY)
save_figure("exp2_sparcity")
plt.show()
_ = plot_mutation_rates(df_split, C_CYCLE, C_SIMILARITY)
save_figure("exp2_similarity")
plt.show()
_ = plot_mutation_rates(df_split, C_CYCLE, C_DELTA)
save_figure("exp2_delta")
plt.show()
# %%
df_ranked = df_grouped.loc[last_cycle_grouped].sort_values([C_VIABILITY])[[C_MODEL, C_SIMULATION_NAME, C_VIABILITY]]
df_ranked[C_RANK] = df_ranked[C_VIABILITY].rank(ascending=False).astype(int)
df_ranked = df_ranked.rename(columns={C_VIABILITY: C_MEAN_VIABILITY})
df_ranked
# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on=C_SIMULATION_NAME, right_on=C_SIMULATION_NAME).sort_values([C_RANK])
df_grouped_ranked
# %%
best_indices = df_grouped_ranked[C_RANK] <= topk
worst_indices = df_grouped_ranked[C_RANK] > df_grouped_ranked[C_RANK].max() - topk
df_grouped_ranked[C_POSITION] = "N/A"
df_grouped_ranked.loc[best_indices, C_POSITION] = f"top{topk}"
df_grouped_ranked.loc[worst_indices, C_POSITION] = f"bottom{topk}"
edge_indices = df_grouped_ranked[C_POSITION] != "N/A"
df_grouped_ranked[C_MRATE] = "" + df_grouped_ranked[C_DELETE].apply(lambda x: f"D={x:.3f}") + " " + df_grouped_ranked[C_INSERT].apply(
    lambda x: f"I={x:.3f}") + " " + df_grouped_ranked[C_CHANGE].apply(lambda x: f"C={x:.3f}")
# df_grouped_ranked[C_MRATE]
df_edge_cases = df_grouped_ranked[edge_indices]
df_edge_cases
# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_edge_cases, x=C_CYCLE, y=C_VIABILITY, ax=faxes, hue=C_MRATE, style=C_POSITION)
# axes.set_xlabel("Evolution Cycle")
axes.set_ylabel("Mean Viability of the current Population")
plt.show()

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
faxes = axes.flatten()
ax = sns.lineplot(data=df_edge_cases, x=C_CYCLE, y=C_FEASIBILITY, ax=faxes[0], hue=C_MRATE, ci=None, legend=False)
# ax.set_xlabel(f"{C_DELETE}: {x_label}")
ax = sns.lineplot(data=df_edge_cases, x=C_CYCLE, y=C_SPARCITY, ax=faxes[1], hue=C_MRATE, ci=None, legend=False)
# ax.set_xlabel(f"{C_INSERT}: {x_label}")
ax = sns.lineplot(data=df_edge_cases, x=C_CYCLE, y=C_SIMILARITY, ax=faxes[2], hue=C_MRATE, ci=None, legend=True)
# ax.set_xlabel(f"{C_CHANGE}: {x_label}")
# ax = sns.lineplot(data=df_edge_cases, x=C_CYCLE, y="delta", ax=faxes[3], hue=C_MRATE, ci=None, legend=True)

fig.tight_layout()

# Steady optimization all diversity is gone
# df[[C_CYCLE,'row.n_population', 'row.n_selection', 'row.n_offspring',
#        'row.n_mutated']].groupby(C_CYCLE).mean()

# %%
# df_subset = df_split.groupby(["id"]).tail(1).reset_index()
# tmp1 = df_subset[C_VIABILITY] > df_subset[C_VIABILITY].quantile(.50)
# df_subset = df_subset.set_index("id")
# df_subset[C_POSITION] = tmp1.values
# df_subset
# # %%
# df_topk = pd.merge(df_split, df_subset.reset_index().loc[:, ["id", C_POSITION]], how="inner", left_on="id", right_on="id")
# df_topk
# #  %%
# fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
# faxes = axes  #.flatten()
# sns.lineplot(data=df_topk[df_topk[C_POSITION]==True], x=x_of_interest, y=C_VIABILITY, ax=faxes, hue="Mutation Rate", ci=None)
# axes.set_xlabel("Evolution Cycle")
# axes.set_ylabel("Mean Viability of the current Population")
# plt.show()
# %%

fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked[edge_indices], x=C_CYCLE, y=C_VIABILITY, ax=faxes, hue=C_MRATE, style=C_POSITION)
# sns.lineplot(data=df_grouped_ranked[~edge_indices], x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_MRATE, estimator="median", style=C_POSITION, alpha=0.2, legend=None)
# axes.set_xlabel("Evolution Cycles")
axes.set_ylabel("Mean Viability of the current Population")
save_figure("exp2_effect_on_viability_top10_last10")
plt.show()
# %%
fig, axes = plt.subplots(1, 1, figsize=(8, 20), sharey=True)
faxes = axes  #.flatten()
sns.lineplot(data=df_grouped_ranked, x=C_CYCLE, y=C_VIABILITY, ax=faxes, hue=C_MRATE, style=C_POSITION)
# sns.lineplot(data=df_grouped_ranked[~edge_indices], x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_MRATE, estimator="median", style=C_POSITION, alpha=0.2, legend=None)
# axes.set_xlabel("Evolution Cycles")
axes.set_ylabel("Mean Viability of the current Population")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

# %%
thresh = 0.0
tmp = df_split.groupby([C_SIMULATION_NAME, C_MODEL, C_INSTANCE]).apply(lambda df: df.iloc[-1][C_VIABILITY] - df.iloc[12][C_VIABILITY]).reset_index()
tmp["viability_rolled"] = tmp[0]
tmp2 = pd.merge(df_split, tmp, how="left", left_on=[C_SIMULATION_NAME, C_MODEL, C_INSTANCE], right_on=[C_SIMULATION_NAME, C_MODEL, C_INSTANCE])

tmp3 = tmp2.loc[(tmp2[C_CYCLE] == tmp2[C_CYCLE].max())]
tmp4 = tmp3[(tmp3["viability_rolled"] > thresh)]
tmp5 = tmp3[(tmp3["viability_rolled"] <= thresh)]

haves = pd.merge(df_grouped_ranked, tmp3, left_on=[C_SIMULATION_NAME], right_on=[C_SIMULATION_NAME], how="left")
## %% tmp3.groupby([C_SIMULATION_NAME, "Model", C_CYCLE]).apply(lambda df: df[C_VIABILITY])
fig, axes = plt.subplots(2, 1, figsize=(10, 15), sharey=True)
faxes = axes  #.flatten()

tmp6 = tmp2.groupby([C_SIMULATION_NAME, C_MODEL, C_CYCLE]).mean().reset_index()
tmp6[C_MRATE] = "" + tmp6[C_DELETE].apply(lambda x: f"D={x:.3f}") + " " + tmp6[C_INSERT].apply(lambda x: f"I={x:.3f}") + " " + tmp6[C_CHANGE].apply(
    lambda x: f"C={x:.3f}")


col_rising = "Has Converged"
tmp6[col_rising] = tmp6["viability_rolled"] <= thresh
tmp6[C_RANGE_DELETE] = pd.cut(tmp6[C_DELETE], bins=bins)
tmp6[C_RANGE_INSERT] = pd.cut(tmp6[C_INSERT], bins=bins)
tmp6[C_RANGE_CHANGE] = pd.cut(tmp6[C_CHANGE], bins=bins)
sns.lineplot(data=tmp6[tmp6[col_rising]], x=C_CYCLE, y=C_VIABILITY, ax=faxes[0], hue=C_MRATE, ci=None)
sns.lineplot(data=tmp6[~tmp6[col_rising]], x=C_CYCLE, y=C_VIABILITY, ax=faxes[1], hue=C_MRATE, ci=None)
for ax in faxes:
    ax.set_ylim(2.3, 2.5)
    # ax.set_xlabel("Evolution Cycles")
    ax.set_ylabel("Mean Viability of the current Population")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
faxes[0].set_title("Converged around 10th Iteration Cycle")
faxes[1].set_title("Still optimizing beyond 10th Iteration Cycle")
# fig.legend(faxes,     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend Title"  # Title for the legend
#            )
fig.tight_layout()
save_figure("sudden_stop_attachment")
plt.show()
# %%
# tmp4[[C_SIMULATION_NAME, "instance", "viability_rolled"]].instance.values
# %%

cols_mrates_classes = [C_DELETE, C_INSERT, C_CHANGE]
# tmp7 = pd.melt(tmp6, id_vars=set(tmp6.columns) - set(cols_mrates_classes), value_vars=cols_mrates_classes)
tmp8 = tmp6[tmp6[C_CYCLE] == tmp6[C_CYCLE].max()]
g = sns.pairplot(data=tmp8, vars=COLS_MRATES+[C_VIABILITY], hue=col_rising)
g.map_lower(sns.histplot)
# g.map_lower(sns.histplot, data=tmp8[~tmp8[col_rising]])

# %%
