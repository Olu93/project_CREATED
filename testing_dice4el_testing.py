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
import pickle
from tqdm import tqdm
# %%
df = pickle.load(open('src/thesis_readers/data/dataset_dice4el/df.pickle', 'rb'))
# %%
df_collector = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    tmp_df = pd.DataFrame([row["activity_vocab"], row["resource_vocab"], row["activity"], row["resource"]]).T
    tmp_df["amount"] = row["amount"]
    tmp_df["caseid"] = row["caseid"]
    
    
    df_collector.extend(tmp_df.values)
    
# %%
cols = ["activity", "resource", "activity_id", "resource_id", "amount", "caseid"]
new_df = pd.DataFrame(df_collector, columns=cols)
new_df
# %%
is_declined = new_df["activity"].str.contains("DECLINE")
is_cancelled = new_df["activity"].str.contains("CANCELLED")
all_failed = new_df.loc[is_declined | is_cancelled, ["caseid"]].values.flatten()
new_df["label"] = 1
new_df.loc[new_df["caseid"].isin(all_failed), "label"] = 0
new_df
# %%
new_df.to_csv("src/thesis_readers/data/dataset_dice4el/dataset_original.csv", index=False)
# %%
