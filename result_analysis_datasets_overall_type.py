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
PATH = pathlib.Path('results/models_overall/datasets/experiment_datasets_overall.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%

df = original_df.copy()
df['iteration'] = df['iteration.no']

df_configs = df  #[df["wrapper_type"] == "EvoGeneratorWrapper"]
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
C_SEPSIS_READER = 'OutcomeSepsisReader'
# %%

df_split = df_configs.copy()
df_split = df_split.rename(columns=map_overall)
df_split[C_SHORT_NAME] = remove_name_artifacts(df_split[C_SHORT_NAME])
df_split = df_split.sort_values([C_MAX_LEN, C_VIABILITY], ascending=False)
df_split[C_VIABILITY] = df_split[COLS_VIAB_COMPONENTS].sum(axis=1)
# df_split[C_MAX_LEN] = df_split[].astype(int)

# df_split = df_split[~df_split[C_EXPERIMENT_NAME].str.contains(C_BPIC_READER)]

# df_split = df_split[df_split["reader"] != "OutcomeSepsisReader100"]


# %%
df_split_grouped = df_split.groupby([
    C_EXPERIMENT_NAME,
    C_SHORT_NAME,
]).median()[[C_VIABILITY, C_MAX_LEN] + COLS_VIAB_COMPONENTS]
df_split_grouped[C_MAX_LEN] = df_split_grouped[C_MAX_LEN].astype(int)
df_split_grouped
# %%
df_table = df_split_grouped#.sort_values(C_VIABILITY, ascending=False)
df_styled = df_table.style.format(
    # escape='latex',
    precision=5,
    na_rep='',
    thousands=" ",
) #.hide(None)
df_latex = df_styled.to_latex(
    multicol_align='l',
    multirow_align='t',
    clines='skip-last;data',
    # column_format='l',
    # caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
    # label=f"tbl:example-cf-{'-'.join(config_name)}",
    hrules=True,
)  #.replace("15 214", "")
save_table(df_latex, "exp5-winner-datasets")

df_styled

# %%
df_split = df_split[~df_split[C_SHORT_NAME].str.startswith("OTFGSL")]
# %%
# %%
fix, ax = plt.subplots(1, 1, figsize=(18, 10))
ax = sns.boxplot(data=df_split, x=C_EXPERIMENT_NAME, y=C_VIABILITY, hue=C_SHORT_NAME, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
save_figure("exp5_winner_overall")

# %%
