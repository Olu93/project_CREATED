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
df['iteration'] = df['iteration.no']

df_configs = df #[df["wrapper_type"] == "EvoGeneratorWrapper"]
df_configs


C_MEASURE = "Measure"
C_MEAUSURE_TYPE = "Viability Measure"
C_OPERATOR = "Operator"
C_OPERATOR_TYPE = "Operator Type"
y_of_interest = C_VIABILITY
x_of_interest = C_CYCLE
C_MEAN_EVENT_CNT = "Fraction of Events"
C_MAX_SEQ_LEN = "max_seq_len"

# %%

df_split = df_configs.copy()
df_split = df_split.rename(columns=map_specifics)
df_split[C_MODEL_CONFIG] = df_split[COLS_OPERATORS].replace(map_operator_long2short).apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
non_names = (df_split[C_MODEL_CONFIG].str.split("_", expand=True) == "nan").values.all(axis=1)
df_split.loc[non_names, C_MODEL_CONFIG] = df_split.loc[non_names, C_SHORT_NAME] 
bins = np.linspace(0, 1, 11, endpoint=True)
df_split[C_RANGE_DELETE] = pd.cut(df_split[C_DELETE], bins=bins)
df_split[C_RANGE_INSERT] = pd.cut(df_split[C_INSERT], bins=bins)
df_split[C_RANGE_CHANGE] = pd.cut(df_split[C_CHANGE], bins=bins)
df_split[C_MEAN_EVENT_CNT] = 1 - df_split[C_PAD_RATIO]
last_cycle = df_split[C_CYCLE] == df_split[C_CYCLE].max()
df_split[C_MRATE] = "" + df_split[C_DELETE].apply(lambda x: f"D={x:.3f}") + " " + df_split[C_INSERT].apply(lambda x: f"I={x:.3f}") + " " + df_split[C_CHANGE].apply(
    lambda x: f"C={x:.3f}")

df_split["is_relevant"] = df_split[C_EXPERIMENT_NAME].str.contains("OutcomeBPIC12Reader") 
df_split = df_split[df_split["is_relevant"]]
df_split[C_MAX_SEQ_LEN] = df_split[C_EXPERIMENT_NAME].str.replace("OutcomeBPIC12Reader", "").astype(int)
df_split[C_ID] = df_split[C_EXPERIMENT_NAME] + " - " + df_split[C_MODEL_CONFIG]  
df_split
# %% plot
topk = 5
df_split[C_CYCLE] = df_split[C_CYCLE].fillna(1)
df_grouped = df_split.groupby([C_ID, C_CYCLE]).mean().reset_index()
df_grouped
# %% plot
fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharey=True)
faxes = axes  #.flatten()
# sns.lineplot(data=df_grouped, x=x_of_interest, y=C_VIABILITY, ax=faxes, hue="instance", legend="full")
sns.lineplot(data=df_grouped, x=x_of_interest, y=C_VIABILITY, ax=faxes, hue=C_ID)

# %%
# %%
df_last_cycle = df_grouped.groupby([C_ID]).tail(1).sort_values([C_VIABILITY])
# df_ranked = df_ranked.rename(columns={C_VIABILITY: "mean_viability"})
df_last_cycle

# %%
df_tmp = df_grouped.merge(df_split[[C_ID, C_MODEL_CONFIG]], on=[C_ID])
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
sns.lineplot(data=df_tmp, y=C_VIABILITY, x=C_MAX_SEQ_LEN, hue=C_MODEL_CONFIG)
# save_figure("exp4_winner_figure_side")
# %%
caption = StringIO()
caption.write("shows the result the average Viability for each model.")
df_table = df_last_cycle.groupby(C_MODEL_CONFIG)
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
