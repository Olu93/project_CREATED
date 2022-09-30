# %%
from collections import defaultdict
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
all_accepted_positions_per_case = new_df.groupby("caseid").apply(lambda xf: xf.reset_index().activity=="A_ACCEPTED_COMPLETE").reset_index().groupby("caseid").apply(lambda df: df[df.activity==True])
all_accepted_positions_per_case

#%%
df_collector2 = defaultdict(list)
df2 = new_df.copy()

# for position in new_df["level_1"]:
#     all_longer = new_df.groupby("caseid").head(position)

for i, pos in tqdm(enumerate(all_accepted_positions_per_case["level_1"]), total=len(all_accepted_positions_per_case["level_1"].unique())):
    tmp_df1 = df2.groupby("caseid").head(pos)
    tmp_df2 = all_accepted_positions_per_case[all_accepted_positions_per_case["level_1"] < pos]
    drop = tmp_df1.index.isin(tmp_df2["caseid"])
    selected = tmp_df1[~drop]
    tmp_df1["l_group"] = pos
    tmp_df1["is_accepted"] = False
    tmp_df1.loc[tmp_df1["caseid"].isin(selected["caseid"].unique()), ["is_accepted"]] = True
    df_collector[pos].append(tmp_df1.values)

# {k: pd.DataFrame(v) for k, v in tqdm(df_collector.items())}
        

# %%


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
