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
PATH = pathlib.Path('results/models_overall/distributions/experiment_distributions_results.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%
df = original_df.copy()
df['prob'] = df['row.prob']
df['apprx_prob'] = df['prob'] + np.finfo(float).eps
df['log_prob'] = np.log(df['row.prob'])
df['apprx_log_prob'] = np.log1p(df['row.prob'])
df['binned_prob'] = pd.cut(df['row.prob'], bins=10)
df['origin_type'] = df['iteration.case_origin']
df['case'] = df['iteration.no']
df['config'] = df['instance.ddist.name']
df['t_estimator'] = df['instance.ddist.transition_estimator']
df['e_estimator'] = df['instance.ddist.emission_estimator']
# %%

only_one_of_the_instances = (df['config'] == df['config'].unique()[0])
df_subset = df[(df['origin_type'] == 'true_cases')]
print(metrics.classification_report(df_subset['row.true_outcome'], df_subset['row.pred_outcome']))
# %%
