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

# %%
PATH = pathlib.Path('results/models_overall/predictions/experiment_predictions_results.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%
df = original_df.copy()
df['case'] = df['iteration.no']
df['dataset'] = df['instance.dataset.class_name']
df['predictor'] = df['instance.predictor']
df['subset'] = df['iteration.subset']
# %%

for idx, group in df.groupby(['dataset', 'predictor', 'subset']):
    print(idx)
    print(metrics.classification_report(group['row.true_outcome'], group['row.pred_outcome']))
# %%
