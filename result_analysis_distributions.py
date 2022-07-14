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
f = plt.figure(figsize=(10, 10))
# sns.displot(df, x='row.viability', kind="kde", hue='type')
g = sns.FacetGrid(
    df,
    row="config",
    hue="origin_type",
    height=1.7,
    aspect=4,
)
g.map(sns.kdeplot, "row.viability")

# %%
f = plt.figure(figsize=(10, 10))
g = sns.FacetGrid(
    df,
    row="config",
    hue="origin_type",
    height=1.7,
    aspect=4,
)
g.map(sns.kdeplot, "log_prob")
# %%
# # https://gist.github.com/fredcallaway/707d2d83b240dc0da50d#file-fred-bigrams-L108
# # \exp \frac{1}{N} -\sum_{k=1}^N \log q(A_k | A_{k-1})
# f = plt.figure(figsize=(10, 10))
# sns.histplot(df.sample(100), x='binned_prob', hue='origin_type')
# %%
# df[df['prob']==df['prob'].min()]
# %%
# https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
# https://stats.stackexchange.com/a/82195/361976
tests = []
for idx, df_grp in df.groupby(['t_estimator', 'e_estimator']):
    data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
    data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
    tmp = stats.kstest(data_true, data_pred)
    display(f"KS-Test: {idx} -> {tmp}")
    tests.append({'type':'Kolmogorov-Smirnov', 'config1':idx[0], 'config2':idx[1], 'value':tmp.statistic, 'significance':tmp.pvalue})
    
for idx, df_grp in df.groupby(['t_estimator', 'e_estimator']):
    data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
    data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
    tmp = spatial.distance.euclidean(data_true, data_pred)
    display(f"L2-Distance: {idx} -> {tmp}")
    tests.append({'type':'L2-Distance', 'config1':idx[0], 'config2':idx[1], 'value':tmp})
    
for idx, df_grp in df.groupby(['t_estimator', 'e_estimator']):
    data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
    data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
    tmp = spatial.distance.cityblock(data_true, data_pred)
    display(f"L1-Distance: {idx} -> {tmp}")
    tests.append({'type':'L1-Distance', 'config1':idx[0], 'config2':idx[1], 'value':tmp})

for idx, df_grp in df.groupby(['t_estimator', 'e_estimator']):
    data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
    data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
    tmp = spatial.distance.correlation(data_true, data_pred)
    display(f"Correlation: {idx} -> {tmp}")
    tests.append({'type':'Correlation', 'config1':idx[0], 'config2':idx[1], 'value':tmp})

pd.DataFrame(tests) 
# for idx, df_grp in df.groupby(['t_estimator', 'e_estimator']):
#     data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
#     data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
#     tmp = stats.kstest(data_true, data_pred)
#     display(f"KS-Test: {idx} -> {tmp}")
#     tests.append({'type':'Kolmogorov-Smirnov', 'config1':idx[0], 'config2':idx[1], 'value':tmp.statistic, 'significance':tmp.pvalue})
    

