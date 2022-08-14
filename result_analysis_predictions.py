# %%
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from sklearn import metrics
from IPython.display import display
from scipy import stats
from scipy import spatial
from jupyter_constants import *
# %%
PATH = pathlib.Path('results/models_overall/predictions/experiment_predictions_results.csv')
original_df = pd.read_csv(PATH)
display(original_df.columns)
original_df.head()
# %%
C_DATASET = "Dataset"

df = original_df.copy()
df['case'] = df['iteration.no']
df['dataset'] = df['instance.dataset.class_name']
df['predictor'] = df['instance.predictor']
df['subset'] = df['iteration.subset']


# %%
def transform(x: Dict, idx: List):
    dataset, predictor, subset = idx
    return {**x, "Dataset": dataset, "Subset": subset}


collector = []
for idx, group in df.groupby(['dataset', 'predictor', 'subset']):
    # print(idx)
    tmp = metrics.classification_report(group['row.true_outcome'], group['row.pred_outcome'], output_dict=True)
    # print(tmp)

    collector.append(transform(tmp.get("weighted avg"), idx))

all_results = pd.DataFrame(collector)
all_results
# %%
df_table = all_results.replace(map_ds_names).pivot(values=[
    "precision",
    "recall",
    "f1-score",
    "support",
], columns=["Subset"], index=["Dataset"])

df_styled = df_table.style.format(
    # escape='latex',
    precision=3,
    na_rep='',
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
save_table(df_latex, "dataset-preds")
df_styled

# %%
