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
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('results/models_overall/evolutionary_iterations/experiment_evolutionary_iterations_results.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%
exog = {
    "gen.initiator.type": "initiator",
    "gen.selector.type": "selector",
    "gen.mutator.type": "mutator",
    "gen.recombiner.type": "recombiner",
    "gen.crosser.type": "crosser",
}

renaming = {
    "run.short_name": "short_name",
    "run.full_name": "full_name",
    "gen.max_iter":"num_iter",
    "row.no":"cf_id",
    "iteration.no":"iteration",
    "instance.no":"instance",
    "run.no":"exp",
}

cols_operators = list(exog.values())

df = original_df.rename(columns=exog)
df = df.rename(columns=renaming)
df["group"] = df["full_name"] + "_" + df["num_iter"].map(str) + "_" + df["instance"].map(str)
df["cfg_set"] = df[cols_operators].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
# sm.GLS(original_df["viability"], original_df[exog])
df
# %%


# %%
fig, ax = plt.subplots(1,1, figsize=(15,15))
sns.lineplot(data=df, x="num_iter", y="viability", hue="cfg_set", ax=ax)
# %%
fig, ax = plt.subplots(1,1, figsize=(10,8))
# df_grouped = df.groupby(["num_iter", "rank"]).mean().reset_index().sort_values("rank")
# df_grouped = df_grouped.groupby(["num_iter", "viability"]).mean().reset_index()
# df_grouped[df_grouped["num_iter"]]
sns.lineplot(data=df, x="num_iter", y="run.duration_sec", hue="iteration")
# sns.lineplot(data=df, x="num_iter", y="viability")
# ax.invert_xaxis()
# ax.set_ylim(0,4)
fig.tight_layout()
# %%
