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


cols_operators = list(map_operators.values())[:3] + list(map_operators.values())[4:]
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = "viability"
x_of_interest = "cycle"
# %%
df_split = df
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
# df_tmp = df_split[df_split['initiator'] != 'FI']
df_tmp = df_split.copy()  
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
    # save_figure(f"exp1_{row}")
    plt.show()

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
# fig, axes = plt.subplots(figsize=(12, 8), sharey=True)
tmp1 = df_split.groupby(['Model']).tail(1).set_index('Model')["viability"] > df_split.groupby(['Model']).tail(1)["viability"].quantile(.75)
models = tmp1.index[tmp1] 
# %%
df_specific = df_split[df_split["Model"].isin(models)]
# df_specific = df_split

plt.figure(figsize=(12, 8))
sns.lineplot(data=df_specific, x=x_of_interest, y="viability", hue="Model", ci=None)
plt.show()
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_specific, x=x_of_interest, y="feasibility", hue="Model", ci=None)
plt.show()
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_specific, x=x_of_interest, y="sparcity", hue="Model", ci=None)
plt.show()
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_specific, x=x_of_interest, y="similarity", hue="Model", ci=None)
plt.show()
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_specific, x=x_of_interest, y="delta", hue="Model", ci=None)
plt.show()

# %%

# %%
