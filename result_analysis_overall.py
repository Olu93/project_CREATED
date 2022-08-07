# %%
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from jupyter_constants import *
from IPython.display import display
# %%
# PATH = pathlib.Path('results/models_overall/overall_sidequest/experiment_overall_sidequest_results.csv')
PATH = pathlib.Path('results/models_overall/overall/experiment_overall_results.csv')
original_df = pd.read_csv(PATH)
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
C_MAX_SEQ_LEN = "Max. Sequence Length"
C_BPIC_READER = 'OutcomeBPIC12Reader'
# %%

df_split = df_configs.copy()
df_split = df_split.rename(columns=map_overall)
df_split[C_SHORT_NAME] = remove_name_artifacts(df_split[C_SHORT_NAME])
df_split

# %%
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))
ax = sns.boxplot(data=df_split, x=C_SHORT_NAME, y=C_VIABILITY, ax =ax1)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, ha="center")
# ax.set_xlabel()
ax = sns.barplot(data=df_split, x=C_SHORT_NAME, y=C_DURATION, ax =ax2)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, ha="center")
fig.tight_layout()
save_figure("exp4_winner_overall")
# %%

caption = StringIO()
caption.write("Table shows the result of Experiment 4. The colors indicate the model configurations that were examined.")
caption.write(" ")
caption.write("The results are based on the average viability each counterfactual a model produces across all factuals that were tested.")

result_table_latex = df_split.groupby(C_SHORT_NAME).mean().iloc[:, :14].style.format(TBL_FORMAT_RULES).to_latex(
    caption=caption.getvalue(),
    label="tbl:exp4-winner",
)
save_table(result_table_latex, "exp4_winner_overall")
display(result_table_latex)

# %%
data = original_df.copy()
all_type_cols = [col for col in data.columns if ("type" in col) and col != "viab.type"]
data["is_degenerate"] = data["cf"].str.split(">").apply(lambda x: np.sum([int(i) for i in x])) == 0
# data["id"] = data["model_name"] + data[all_type_cols].fillna("NA").agg('_'.join, axis=1)
# data["id"].str.extract(r"((([A-Z])+)_)+", expand=True)
# data["shortid"] = data["id"].str.replace(pat=r"([a-z])+", repl="", regex=True)
data["is_correct"] = data["result_outcome"] == data["target_outcome"]
data
# %%
top_10 = data[(data["rank"] < 11)]
top_10_means = top_10.groupby(["run.short_name", "iteration.no"]).mean()
top_10_means["is_correct"] = (top_10_means["is_correct"] == 1)

# %%
# df_tmp = top_10.copy()
# df_intermediate = df_tmp[df_tmp.columns[df_tmp.isnull().sum() == 0]]

# df_melt = df_intermediate.melt(id_vars=set(df_intermediate.columns) - set(COLS_VIAB_COMPONENTS), value_vars=COLS_VIAB_COMPONENTS, var_name=C_VIABILITY_COMPONENT, value_name="Value")
# sns.relplot(data=df_melt, x="Value", y="viability", hue=C_VIABILITY_COMPONENT)
# %%
# col_selector = (data.model_name != "SimpleGeneratorModel") & (data.model_name != "RandomGenerator") & (data.is_degenerate != True)
# df_tmp = top_10[col_selector].copy()
df_tmp = data.copy().rename(columns=map_parts_overall)
df_intermediate = df_tmp[df_tmp.columns[df_tmp.isnull().sum() == 0]]

df_melt = df_intermediate.melt(id_vars=set(df_intermediate.columns) - set(COLS_VIAB_COMPONENTS), value_vars=COLS_VIAB_COMPONENTS, var_name=C_VIABILITY_COMPONENT, value_name="Value")
sns.pairplot(data=df_intermediate, vars=COLS_VIAB_COMPONENTS+["viability"], hue="run.short_name")
# Second pairplot with different lower triangle. Maybe target outcome
# save_figure("exp4_all_vs_all")
