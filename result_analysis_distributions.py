# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from sklearn import metrics
from IPython.display import display
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
# https://gist.github.com/fredcallaway/707d2d83b240dc0da50d#file-fred-bigrams-L108
# \exp \frac{1}{N} -\sum_{k=1}^N \log q(A_k | A_{k-1})
f = plt.figure(figsize=(10, 10))
sns.histplot(df.sample(100), x='binned_prob', hue='type')
# %%
# df[df['prob']==df['prob'].min()]
# %%
for idx, df_grp in df.groupby(['config']):
    data_true = df_grp[(df_grp['origin_type'] == 'true_cases')]["prob"]
    data_pred = df_grp[(df_grp['origin_type'] == 'sampled_cases')]["prob"]
    tmp = metrics.mutual_info_score(data_true, data_pred)
    display(f"Kl-Divergence: {idx} -> {tmp}")