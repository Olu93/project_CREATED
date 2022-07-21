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
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_iterations_specifics.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%

df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df[df['experiment_name'] == "evolutionary_iterations"]
df_configs
cols_operators = list(map_operators.values())
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
x_of_interest = "relative_duration"
cat_of_interest = 'Max. Number of Cycles'
# %%
renamings = {**map_parts, **map_viability, **map_operators, **map_mrates, **map_erate}
df_split = df_configs.copy()
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2], axis=1)
configurations_full_name = configurations.copy().replace(map_operator_shortnames)
df_split = df_split.join(configurations).rename(columns=renamings)
df_split['Model'] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
df_split['ncycles'] = df_split[9].str.replace("ncycles", "").str.replace(".csv", "").astype(int)
df_split[cat_of_interest] = pd.Categorical(df_split['ncycles'])
# %%

df_split[x_of_interest] = df_split.groupby('ncycles')['cycle'].apply(lambda series: (series-series.min())/(series.max()-series.min()))
df_split
# %%
fig, ax = plt.subplots(1,1, figsize=(10,5))
sns.lineplot(data=df_split, x=x_of_interest, y=y_of_interest, hue=cat_of_interest, ax=ax, ci=None)
plt.legend(title="Num. of Iterative Cycles",bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.tight_layout()
save_figure("exp3_relative_cycles")
# %%
df_last_results = df_split.groupby(['ncycles', 'cycle']).tail(1)
fig, ax = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(data=df_last_results, x=cat_of_interest, y=y_of_interest, hue='Model', ax=ax)
plt.legend(title="Model Configuration", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.tight_layout()

save_figure("exp3_cycles_spread")
# %%
