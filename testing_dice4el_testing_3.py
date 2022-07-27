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
from tqdm.notebook import tqdm
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
df_collector2 = []
df2 = new_df.copy()
df2["is_accepted"] = False
# for acc_cid, acc_df in tqdm(all_accepted_positions_per_case.groupby("caseid"), total=len(all_accepted_positions_per_case["level_1"].unique())):
for pos, accepted in tqdm(all_accepted_positions_per_case.groupby("level_1"), total=len(all_accepted_positions_per_case["level_1"].unique())):
    # sub_set = df2.loc[single_acc_cid in df2["caseid"]]
    # sub_set = df2.loc[:, :pos]
    # df_at_most = df2.groupby("caseid").head(pos).groupby("caseid").tail(1)
    df_at_most = df2.groupby("caseid").head(pos)
    for cid, subgrp in tqdm(df_at_most.groupby("caseid"), total=len(df_at_most["caseid"].unique())):
        if len(subgrp) < pos:
            continue
        subgrp["is_accepted"] = cid in accepted["caseid"]
        subgrp["pos"] = pos
        df_collector2.extend(subgrp.values)
    # caseids_of_accepted = df_at_most[df_at_most["caseid"].isin(accepted["caseid"])]["caseid"]
    # df_tmp = df2.loc[df2["caseid"].isin(caseids_of_accepted)]
    # df_tmp["is_accepted"] = False
    
        

# {k: pd.DataFrame(v) for k, v in tqdm(df_collector.items())}
        

# %%
cols2 = ["activity", "resource", "activity_id", "resource_id", "amount", "caseid", "is_accepted", "pos"]
new_df_2 = pd.DataFrame(df_collector2, columns=cols2)
new_df_2
# %%
# is_declined = new_df["activity"].str.contains("DECLINE")
# is_cancelled = new_df["activity"].str.contains("CANCELLED")
# all_failed = new_df.loc[is_declined | is_cancelled, ["caseid"]].values.flatten()
# new_df["label"] = 1
# new_df.loc[new_df["caseid"].isin(all_failed), "label"] = 0
# new_df
# # %%
# new_df.to_csv("src/thesis_readers/data/dataset_dice4el/dataset_original.csv", index=False)
# # %%
