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
C_SEPSIS_READER = 'OutcomeSepsisReader'
# %%

df_split = df_configs.copy()
df_split = df_split.rename(columns=map_overall)
# df_split = df_split[~df_split[C_EXPERIMENT_NAME].str.contains(C_BPIC_READER)] 

df_split = df_split[df_split["reader"] != "OutcomeSepsisReader100"]
df_split
# %%
df_split = df_split[~df_split[C_SHORT_NAME].str.startswith("OTFGSL")]
# %%
df_split.groupby([C_SHORT_NAME, C_EXPERIMENT_NAME]).mean()
# %%
fix, ax = plt.subplots(1,1, figsize=(18,10))
ax = sns.boxplot(data=df_split, x=C_EXPERIMENT_NAME, y=C_VIABILITY, hue=C_SHORT_NAME, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
save_figure("exp5_winner_overall")

# %%
