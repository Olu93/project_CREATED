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
PATH = pathlib.Path('results/models_specific/grouped_evolutionary_sidequest_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%


df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df
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

# %%
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
    # save_figure(f"sidequest_{row}")
    plt.show()
# %%
EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_TKS_TPC_DDM_PR_IM.csv')
EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_USS_TPC_DDM_PR_IM.csv')
# EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_ES_TPC_DDM_PR_IM.csv')
evo_df = pd.read_csv(EVO_PATH)
evo_df.head(10)
# %%
evo_df_means = evo_df.groupby(['instance.no', 'iteration.no']).mean().reset_index()
evo_df_means


# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_sparcity", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_similarity", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_feasibility", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_delta", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_viability", hue="instance.no")
# %%
# sns.ecdfplot(data=evo_df_means, x="viability")
# %%
