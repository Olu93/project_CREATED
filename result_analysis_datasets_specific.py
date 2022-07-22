# %%
from io import StringIO
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
# PATH = pathlib.Path('results/models_specific/grouped_overall_specifics.csv')
PATH = pathlib.Path('results/models_specific/grouped_datasets_specifics.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%

df = original_df.copy()
df['model'] = df['filename']
df['instance'] = df['instance.no']
df['iteration'] = df['iteration.no']
df['cycle'] = df['row.num_cycle']

df_configs = df[df["wrapper_type"] == "EvoGeneratorWrapper"]
df_configs
cols_operators = list(map_operators.values())[:3] + list(map_operators.values())[4:]
cols_parts = list(map_parts.values())

C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = C_VIABILITY
x_of_interest = "cycle"
C_MEAN_EVENT_CNT = "Fraction of Events"

# %%
renamings = {**map_parts, **map_viability, **map_operators, **map_mrates, **map_erate, **map_viability_overall}
df_split = df_configs.copy()
configurations = df_split["model"].str.split("_", expand=True).drop([0, 1, 2], axis=1)
configurations_full_name = configurations.copy().replace(map_operator_shortnames)
df_split = df_split.join(configurations).rename(columns=renamings)
df_split[C_MODEL_CONFIG] = df_split[cols_operators].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
bins = np.linspace(0, 1, 11, endpoint=True)
df_split[C_RANGE_DELETE] = pd.cut(df_split[C_DELETE], bins=bins)
df_split[C_RANGE_INSERT] = pd.cut(df_split[C_INSERT], bins=bins)
df_split[C_RANGE_CHANGE] = pd.cut(df_split[C_CHANGE], bins=bins)
df_split[C_MEAN_EVENT_CNT] = 1 - df_split["iteration.mean_num_zeros"]
last_cycle = df_split["cycle"] == df_split["cycle"].max()
df_split["Mutation Rate"] = "" + df_split[C_DELETE].apply(lambda x: f"D={x:.3f}") + " " + df_split[C_INSERT].apply(lambda x: f"I={x:.3f}") + " " + df_split[C_CHANGE].apply(
    lambda x: f"C={x:.3f}")
# %% plot
topk = 5
df_grouped = df_split.groupby([C_MODEL_CONFIG, "cycle", "instance"]).mean().reset_index()
last_cycle_grouped = df_grouped["cycle"] == df_grouped["cycle"].max()
# %% plot
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
# sns.lineplot(data=df_grouped, x=x_of_interest, y=C_VIABILITY, ax=faxes, hue="instance", legend="full")
sns.lineplot(data=df_grouped, x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_MODEL_CONFIG)

# %%
# %%
df_ranked = df_grouped.loc[last_cycle_grouped].sort_values([C_VIABILITY])
df_ranked["rank"] = df_ranked.groupby(C_MODEL_CONFIG).apply(lambda df: df[C_VIABILITY].rank(ascending=False).astype(int)).values
# df_ranked = df_ranked.rename(columns={C_VIABILITY: "mean_viability"})
df_ranked
# %%
df_grouped_ranked = pd.merge(df_grouped, df_ranked, how="inner", left_on=[
    C_MODEL_CONFIG,
], right_on=[
    C_MODEL_CONFIG,
]).sort_values(["rank"])
df_grouped_ranked
# %%
df_rival = pd.read_csv(PATH)
df_rival = df_rival[df_rival["wrapper_type"] != "EvoGeneratorWrapper"]
df_rival[C_MODEL_CONFIG] = None
df_rival[C_MODEL_CONFIG] = df_rival["wrapper_type"]
df_rival = df_rival.rename(columns=renamings)

# for idx, df in
df_rival["rank"] = df_rival.groupby(C_MODEL_CONFIG).apply(lambda df: df[C_VIABILITY].rank(ascending=False).astype(int)).values
df_rival
# %%
df_combined = pd.concat([df_ranked, df_rival])
df_combined
# %%
keep_cols = df_combined.columns[df_combined.isnull().sum() == 0]
df_aligned = df_combined[keep_cols].drop("Unnamed: 0", axis=1).rename(columns=renamings)
df_aligned = df_aligned.sort_values([C_MODEL_CONFIG, "rank"])
df_aligned[[C_MODEL_CONFIG, "rank"]]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(data=df_aligned, x="rank", y=C_VIABILITY, hue=C_MODEL_CONFIG)
ax.invert_xaxis()
# %%
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
sns.boxplot(data=df_aligned, y=C_VIABILITY, x=C_MODEL_CONFIG)
# save_figure("exp4_winner_figure_side")
# %%
caption = StringIO()
caption.write("shows the result the average Viability for each model.")
df_table = df_aligned.groupby(C_MODEL_CONFIG)
result_table_latex1 = df_table[COLS_VIAB_COMPONENTS + [C_VIABILITY]].mean().style.format(escape='latex').to_latex(
    caption=caption.getvalue(),
    label="tbl:exp4-winner1",
).replace("_", "-")
print(result_table_latex1)
# %%
result_table_latex2 = df_table[COLS_VIAB_COMPONENTS + [C_VIABILITY]].std().style.format(escape='latex').to_latex(
    caption=caption.getvalue(),
    label="tbl:exp4-winner2",
).replace("_", "-")
print(result_table_latex2)
# %%
# save_table(result_table_latex1, "exp4-winner1")
# %%
