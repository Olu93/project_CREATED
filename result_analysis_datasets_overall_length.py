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
df_split = df_split.drop([c for c in df_split.columns if "gen.SeqProcessLoss" in c], axis=1).drop([c for c in df_split.columns if "gen.SeqProcessEvaluator" in c], axis=1)
tmp = pd.DataFrame([df_split[C_EXPERIMENT_NAME].str.contains(maxlen).values for maxlen in ["25", "50", "75", "100"]])
df_split = df_split[tmp.any(axis=0).values] 
df_split[C_MAX_SEQ_LEN] = df_split[C_EXPERIMENT_NAME].str.replace(C_BPIC_READER, "").replace(C_SEPSIS_READER, "").astype(int)
df_split

# %%
fix, ax = plt.subplots(1,1, figsize=(10,10))
sns.boxplot(data=df_split, x=C_MAX_SEQ_LEN, y=C_VIABILITY, hue=C_SHORT_NAME)
# %%
fix, ax = plt.subplots(1,1, figsize=(10,10))
sns.lineplot(data=df_split, x=C_MAX_SEQ_LEN, y=C_DURATION, hue=C_SHORT_NAME)
# %%

df_split.groupby([C_SHORT_NAME, C_EXPERIMENT_NAME]).mean()