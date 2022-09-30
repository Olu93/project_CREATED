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
    df_collector.append(pd.DataFrame(tmp_df))
    
# %%
cols = ["activity", "resource", "activity_id", "resource_id", "amount", "caseid"]
new_df = pd.concat(df_collector)
new_df.columns = cols
new_df
# %%
# all_accepted_positions_per_case = new_df.groupby("caseid").apply(lambda xf: xf.reset_index().activity=="A_ACCEPTED_COMPLETE").reset_index().groupby("caseid").apply(lambda df: df[df.activity==True])
# all_accepted_positions_per_case
selector = ['A_CANCELLED_COMPLETE', 'A_DECLINED_COMPLETE']
all_declined_pos = new_df.groupby("caseid").apply(lambda xf: xf.reset_index().activity.isin(selector)).reset_index().groupby("caseid").apply(lambda df: df[df.activity==True])
# pd.concat([all_declined_pos, all_canceled_pos])
all_declined_pos
# %%
labelled_df = new_df.merge(all_declined_pos.reset_index(drop=True).rename(columns={"level_1":"pos", "activity":"label"}), on="caseid", how='left').fillna(False)
labelled_df.loc[labelled_df["label"]==True, ["label"]] = "deviant"
labelled_df.loc[labelled_df["label"]==False, ["label"]] = "regular"

labelled_df
# %%
labelled_df.to_csv("src/thesis_readers/data/dataset_dice4el/labelled_df.csv")


# #%%
# df_collector2 = []
# df_collector_tmp = []
# new_df_2 = pd.DataFrame()
# df2 = new_df.copy()

# df2["is_declined"] = False
# # for acc_cid, acc_df in tqdm(all_accepted_positions_per_case.groupby("caseid"), total=len(all_accepted_positions_per_case["level_1"].unique())):
# for pos, declined in tqdm(all_declined_pos.groupby("level_1"), total=len(all_declined_pos["level_1"].unique())):
#     # sub_set = df2.loc[single_acc_cid in df2["caseid"]]
#     # sub_set = df2.loc[:, :pos]
#     # df_at_most = df2.groupby("caseid").head(pos).groupby("caseid").tail(1)
#     df_at_most = df2.groupby("caseid").head(pos+1)
#     for cid, subgrp in tqdm(df_at_most.groupby("caseid"), total=len(df_at_most["caseid"].unique())):
#         if len(subgrp) < pos+1:
#             continue
#         subgrp["is_declined"] = (cid in declined["caseid"])
#         subgrp["pos"] = pos+1
#         subgrp["group"] = subgrp.tail(1).activity
#         # new_df_2 = pd.concat([new_df_2, subgrp.values[:-1]], ignore_index=True)
#         df_collector2.append(pd.DataFrame(subgrp.values[:-1]))
#     # caseids_of_accepted = df_at_most[df_at_most["caseid"].isin(accepted["caseid"])]["caseid"]
#     # df_tmp = df2.loc[df2["caseid"].isin(caseids_of_accepted)]
#     # df_tmp["is_accepted"] = False
    
        

# # {k: pd.DataFrame(v) for k, v in tqdm(df_collector.items())}
        

# # %%
# cols2 = ["activity", "resource", "activity_id", "resource_id", "amount", "caseid", "is_declined", "pos", "drop"]
# new_df_2 = pd.concat(df_collector2)
# new_df_2.columns = cols2
# new_df_2
# # %%
# # is_declined = new_df["activity"].str.contains("DECLINE")
# # is_cancelled = new_df["activity"].str.contains("CANCELLED")
# # all_failed = new_df.loc[is_declined | is_cancelled, ["caseid"]].values.flatten()
# # new_df["label"] = 1
# # new_df.loc[new_df["caseid"].isin(all_failed), "label"] = 0
# # new_df
# # # %%
# # new_df.to_csv("src/thesis_readers/data/dataset_dice4el/dataset_original.csv", index=False)
# # # %%
# # new_df_2.loc[new_df_2["caseid"].isin([212548])]
# # %%
# all_declined_pos
# # %%
# new_df.reset_index()[new_df.reset_index().caseid=="173694"]
# # %%

# %%
