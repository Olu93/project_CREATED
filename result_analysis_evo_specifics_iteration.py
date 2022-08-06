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
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_iterations_specifics.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%

df = original_df.copy()

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
x_of_interest = "relative_duration"
# %%
# %%
df_split = df.copy()
df_split = df_split.rename(columns=map_specifics)
# df_split[C_CYCLE_TERMINATION] = df_split[C_SIMULATION_NAME].str.split("ncycles", expand=True).iloc[:, -1].str.replace(".csv", "").astype(int)
# df_split[C_CYCLE_TERMINATION] = pd.Categorical(df_split[C_CYCLE_TERMINATION])
df_split[C_SHORT_NAME] = df_split[C_SHORT_NAME].str.replace("ES_EGW_", "").str.replace("_IM", "").str.replace("_", "-")
df_split[C_EVT_RATIO] = 1-df_split[C_PAD_RATIO]
df_split

# df_split[C_CYCLE_NORMED] = df_split.groupby([C_CYCLE_TERMINATION])[C_CYCLE].apply(lambda series: (series-series.min())/(series.max()-series.min()))
# df_split
# %%
fig, ax = plt.subplots(1,1, figsize=(12,6))
sns.lineplot(data=df_split, x=C_CYCLE, y=C_VIABILITY, hue=C_SHORT_NAME, ax=ax, ci=None, legend='full')
# plt.legend(title=C_MODEL_CONFIG, bbox_to_anchor=(1.02, 0.75), loc=2, borderaxespad=0.)
fig.tight_layout()
save_figure("exp3_relative_cycles")
# %%
num_comp = len(COLS_VIAB_COMPONENTS)
fig, axes = plt.subplots(2,3, figsize=(12,8))
faxes = axes.flatten()
for measure, ax in zip(COLS_VIAB_COMPONENTS, faxes[:num_comp]):
    ax = sns.lineplot(data=df_split, x=C_CYCLE, y=measure, hue=C_SHORT_NAME, ax=ax, ci=None, legend=measure==C_FEASIBILITY)
    ax.set_ylim(0,1)
# ax.legend(title=C_MODEL_CONFIG, bbox_to_anchor=(1.02, 1.2), loc=1, borderaxespad=0.)
    
for measure, ax in zip([C_EVT_RATIO,C_FEASIBILITY], faxes[num_comp:]):
    # if measure in [C_EVT_RATIO]:
    #     sns.lineplot(data=df_split, x=C_CYCLE, y=measure, hue=C_SHORT_NAME, ax=ax, ci=None, legend=None)
    #     ax.set_ylim(0,1)
    if measure in [C_EVT_RATIO]:
        sns.lineplot(data=df_split, x=C_CYCLE, y=measure, hue=C_SHORT_NAME, ax=ax, ci=None, legend=None)
        ax.set_ylim(0,1)
    if measure in [C_FEASIBILITY]:
        ax = sns.lineplot(data=df_split, x=C_CYCLE, y=measure, hue=C_SHORT_NAME, ax=ax, ci=None, legend=None)
        ax.set_ylabel(C_FEASIBILITY + f" (Modified Y-Axis)")
fig.tight_layout()
save_figure("exp3_cycles_components")

# df_last_results = df_split.groupby([C_CYCLE_TERMINATION, C_CYCLE]).tail(1)
# fig, ax = plt.subplots(1,1, figsize=(10,5))
# ax = sns.boxplot(data=df_last_results, x=C_CYCLE_TERMINATION, y=C_VIABILITY, ax=ax)
# low_point = df_last_results.groupby(C_CYCLE_TERMINATION)[C_VIABILITY].min().median()
# ax.axhline(y=low_point, color='red', linestyle = '-.', label="Low Point")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# fig.tight_layout()
# save_figure("exp3_cycles_spread")
# %%