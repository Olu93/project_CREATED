# %%
import datetime
import io
import os
import pathlib
import sys
import traceback
import itertools as it
from typing import Union, final
import tensorflow as tf
from thesis_readers.readers.OutcomeReader import OutcomeDice4ELEvalReader, OutcomeDice4ELReader, OutcomeTrafficShortReader
from functools import lru_cache

keras = tf.keras
from tqdm import tqdm
import time
from thesis_commons.constants import (PATH_PAPER_COUNTERFACTUALS, PATH_PAPER_FIGURES, PATH_PAPER_TABLES, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, CDType)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc
import seaborn as sns
from IPython.display import display

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
# pd.set_option("chop", 10)
#%%
task_mode = TaskModes.OUTCOME_PREDEFINED
ft_mode = FeatureModes.FULL
epochs = 5
batch_size = 32
ff_dim = 5
embed_dim = 5
adam_init = 0.1
# sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
sample_size = 200
num_survivors = 1000
experiment_name = "dice4el"
outcome_of_interest = None

ds_name = "OutcomeTrafficShortReader"
reader: OutcomeTrafficShortReader = OutcomeTrafficShortReader.load(PATH_READERS / ds_name)
reader.original_data[reader.original_data[reader.col_case_id] == 173688]

# %% ------------------------------
(events, features), labels = reader.get_dataset_example(ds_mode=DatasetModes.ALL, ft_mode=FeatureModes.FULL)
reader.decode_matrix_str(events)


def save_figure(title: str):
    plt.savefig((PATH_PAPER_FIGURES / title).absolute(), bbox_inches="tight")


def save_table(table: Union[str, pd.DataFrame], filename: str):
    if isinstance(table, pd.DataFrame):
        table = table.style.format(escape="latex").to_latex()
    destination = PATH_PAPER_COUNTERFACTUALS / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table.replace("_", "-"))


# %%
reader.decode_results(events, features, labels)
# %%

list_of_cfs = pickle.load(io.open(PATH_RESULTS_MODELS_OVERALL / experiment_name / 'results_traffic.pkl', 'rb'))
list_of_cfs

# %%
# def convert_to_dice4el_format(reader, df_post_fa, prefix=""):
#     convert_to_dice4el_format = df_post_fa.groupby("id").apply(
#         lambda x: {
#             prefix + '_' + 'amount': list(x["AMOUNT_REQ"]),
#             prefix + '_' + 'activity': list(x[reader.col_activity_id]),
#             prefix + '_' + 'resource': list(x[C_RESOURCE]),
#             # prefix + '_' + 'feasibility': list(x.feasibility)[0],
#             prefix + '_' + 'label': list(x.label),
#             prefix + '_' + 'id': list(x.id)[0]
#         }).to_dict()
#     sml = pd.DataFrame(convert_to_dice4el_format).T.reset_index(drop=True)
#     return sml
C_G = 'G'

G_CASE = (C_G, 'case')
G_NAME = (C_G, 'name')
G_ITER = (C_G, 'iteration')
G_MODL = (C_G, 'model_num')
G_STEP = (C_G, 'step')
C_FA = 'FA'
C_CF = 'CF'
# C_D4 = 'D4EL'
COL_MAPPER = {C_FA: "Factual Seq.", C_CF: "Our CF Seq."}

C_READER_ACTIVITY = reader.col_activity_id
C_READER_AMOUNT = "open_cases"
C_READER_RESOURCE = "Resource"
C_READER_OUTCOME = reader.col_outcome

C_ACTIVITY = 'Activity'
C_RESOURCE = 'Resource'
C_AMOUNT = 'Open Cases'
C_OUTCOME = 'Outcome'

C_NAME_MAPPER = {C_READER_ACTIVITY: C_ACTIVITY, C_READER_RESOURCE: C_RESOURCE, C_READER_AMOUNT: C_AMOUNT, C_READER_OUTCOME: C_OUTCOME}
# GLUE_GROUPS = [C_FA, C_CF, C_D4]
GLUE_GROUPS = [C_FA, C_CF]
GLUE_COLS = [G_CASE, G_NAME, G_ITER, G_MODL, G_STEP]
# GLUE_COLS = [G_NAME, G_ITER, G_MODL, G_STEP]
# GLUE_TAIL_GROUPER = list([tuple(els) for els in np.array(GLUE_COLS)[[1, 2, 4]]])
GLUE_TAIL_GROUPER = [G_NAME, G_ITER, G_STEP]
GLUE_GID = [G_NAME, G_CASE, G_ITER]
# GLUE_GID = [G_NAME, G_ITER]
# GLUE_TAKER = list([tuple(els) for els in np.array(GLUE_COLS)[[1, 2]]])
GLUE_TAKER = [G_NAME, G_ITER]
GLUE_DISPLAY = [
    C_ACTIVITY,
    C_RESOURCE,
    C_AMOUNT,
    C_OUTCOME,
]
# %%
big_collector = []
collector = []
for model_num, el in enumerate(list_of_cfs):
    name, cf, fa, iteration = el.values()
    # for result_id, (cf, fa) in enumerate(zip(cf_list, fa_list)):
    # for case_id, case in enumerate(cases):

    events, features = cf.cases
    llh = cf.likelihoods
    viabilities = cf.viabilities
    sparcity = viabilities.sparcity
    similarity = viabilities.similarity
    feasibility = viabilities.dllh
    delta = viabilities.ollh
    viability = viabilities.viabs
    cf_df_tmp = reader.decode_results(events, features, llh > 0.5, sparcity, similarity, feasibility, delta, viability)
    old_cols = [col for col in cf_df_tmp.columns if col != 'case']
    new_cols = pd.MultiIndex.from_product([[C_CF], old_cols])
    renamer = dict(zip(old_cols, new_cols))
    renamer["case"] = G_CASE
    df_tmp = pd.DataFrame(cf_df_tmp)  #.rename(columns=renamer)
    df_tmp = df_tmp.rename(columns=renamer)[list(renamer.values())]
    df_tmp.columns = pd.MultiIndex.from_tuples(df_tmp.columns)
    df_tmp[G_NAME] = name
    df_tmp[G_CASE] = cf_df_tmp["case"]#.astype(int)
    df_tmp[G_ITER] = iteration
    df_tmp[G_MODL] = model_num
    collector.append(df_tmp)
all_results = pd.concat(collector)
all_results[GLUE_COLS[-1]] = all_results.groupby(GLUE_COLS[:-1]).cumcount()
all_results
# %%
collector = []
for model_num, el in enumerate(list_of_cfs):
    name, cf, fa, iteration = el.values()
    events, features = fa.cases
    llh = fa.likelihoods
    fa_df_tmp = reader.decode_results(events, features, llh > 0.5)
    old_cols = [col for col in fa_df_tmp.columns if col != 'case']
    new_cols = pd.MultiIndex.from_product([[C_FA], old_cols])
    renamer = dict(zip(old_cols, new_cols))
    renamer["case"] = G_CASE
    df_tmp = pd.DataFrame(fa_df_tmp)  #.rename(columns=renamer)
    df_tmp = df_tmp.rename(columns=renamer)[list(renamer.values())]
    df_tmp.columns = pd.MultiIndex.from_tuples(df_tmp.columns)
    df_tmp[G_NAME] = name
    df_tmp[G_CASE] = fa_df_tmp["case"]#.astype(int)
    df_tmp[G_ITER] = iteration
    df_tmp[G_MODL] = model_num
    collector.append(df_tmp)
fa_all_results = pd.concat(collector)
fa_all_results[GLUE_COLS[-1]] = fa_all_results.groupby(GLUE_COLS[:-1]).cumcount()
fa_all_results

# %%
# X, Y = reader.get_d4el_result_cfs(is_df = False)
# X[1]


def pad_length(x, to_length, padding_value=None):
    return ([padding_value] * (to_length - len(x))) + x


def expand_d4el(reader, df):
    cols_2_convert = ['activity_vocab', 'resource_vocab']
    cols_remaining = list(set(df.columns) - set(cols_2_convert))
    df[cols_2_convert[0]] = [pad_length(df[cols_2_convert[0]].values[0], reader.max_len, reader.pad_token)]
    df[cols_2_convert[1]] = [pad_length(df[cols_2_convert[1]].values[0], reader.max_len, reader.pad_token)]

    df2 = df[cols_2_convert].apply(pd.Series.explode).join(df[cols_remaining])
    df2 = df2.drop(['activity', 'resource'], axis=1)
    df2 = df2.rename(columns={
        'activity_vocab': C_READER_ACTIVITY,
        'resource_vocab': C_READER_RESOURCE,
        'amount': C_READER_AMOUNT,
    })

    return df2


# collector = []
# d4el_df = reader._d4e_cfs.groupby("caseid_seed").head(1)
# for model_num, el in enumerate(list_of_cfs):
#     name, cf, fa, iteration = el.values()
#     d4el_df_tmp = expand_d4el(reader, d4el_df[iteration:iteration + 1])
#     d4el_df_tmp[C_READER_ACTIVITY] = d4el_df_tmp[C_READER_ACTIVITY].str.replace("_COMPLETE", "")
#     # d4el_df_tmp.columns = pd.MultiIndex.from_product([[C_D4], d4el_df_tmp.columns])
#     old_cols = [col for col in d4el_df_tmp.columns if col != 'case']
#     new_cols = pd.MultiIndex.from_product([[C_D4], old_cols])
#     renamer = dict(zip(old_cols, new_cols))
#     # renamer["case"] = G_CASE
#     df_tmp = pd.DataFrame(d4el_df_tmp)  #.rename(columns=renamer)
#     df_tmp = df_tmp.rename(columns=renamer)[list(renamer.values())]
#     df_tmp.columns = pd.MultiIndex.from_tuples(df_tmp.columns)
#     df_tmp[G_NAME] = name
#     df_tmp[G_CASE] = None
#     df_tmp[G_ITER] = iteration
#     df_tmp[G_MODL] = model_num
#     collector.append(df_tmp)
# d4el_all_results = pd.concat(collector)
# d4el_all_results[GLUE_COLS[-1]] = d4el_all_results.groupby(GLUE_COLS[1:-1]).cumcount()
# d4el_all_results
# %%
df_merged = all_results.copy()
df_merged = df_merged.merge(fa_all_results, how='left', on=GLUE_COLS)
# df_merged = df_merged.merge(d4el_all_results, how='left', suffixes=(None, "_2"), on=GLUE_COLS[1:3] + GLUE_COLS[4:])
df_merged["gid"] = df_merged[GLUE_GID].astype(str).apply('_'.join, axis=1)
df_merged

df_merged = df_merged.set_index("gid")
df_merged = df_merged.rename(columns=C_NAME_MAPPER, level=1)

df_merged.loc(axis=1)[:, C_RESOURCE] = df_merged.loc(axis=1)[:, C_RESOURCE].astype(str).apply(lambda x: x.str.replace('.0', '').str.replace('UNKNOWN', 'other')).values
# df_merged.loc(axis=1)[:, C_AMOUNT] = np.floor(df_merged.loc(axis=1)[:, C_AMOUNT]).astype(str).apply(lambda x: x.str.replace('.0', '')).values
# df_merged.loc(axis=1)[:, C_OUTCOME] = np.floor(df_merged.loc(axis=1)[:, C_OUTCOME]).astype(str).apply(lambda x: x.str.replace('.0', '')).values
df_merged


# %%
def generate_latex_table(df: pd.DataFrame, index, suffix="", caption=""):

    df = df.loc[:, GLUE_GROUPS]

    # Remove all lines that are NaN
    df = df[df.notnull().any(axis=1)]
    # Take relevant subcolumns
    df = df.loc[:, df.columns.get_level_values(1).isin(GLUE_DISPLAY)]

    config_name = f"{suffix}-" + index.replace("_", "-")  

    # Remove all unimportant data
    mapper = COL_MAPPER
    for e in mapper.keys():
        del_index = df[(e, C_ACTIVITY)].isin(["<s>", "</s>", "nan", "<UNK>"])
        df.loc[del_index, e] = ""
    df = df.rename(columns=mapper, level=0)
    df = df.replace("nan", "").replace(np.nan, "").replace("<PAD>", "").replace("<SOS>", "").replace("<EOS>", "")  

    # Remove all empty lines
    df = df[~np.all(df=="", axis=1)]

    df_styled = df.style.format(
        # escape='latex',
        precision=0,
        na_rep='',
        thousands=" ",
    ).hide(None)
    df_latex = df_styled.to_latex(
        multicol_align='l',
        multirow_align='t',
        clines='skip-last;data',

        # column_format='l',
        # caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
        # label=f"tbl:example-cf-{'-'.join(config_name)}",
        hrules=True,
    )  #.replace("15 214", "")
    return df, df_styled, df_latex, config_name


# all_results.groupby(["G_model_num", "G_step", "G_iteration"]).tail(1)
with io.open("traffic_saved_results_counterfactual_general.txt", "w") as file:
    best_ones = df_merged.groupby(GLUE_TAIL_GROUPER).tail(1).reset_index()
    for index, df in best_ones.groupby('gid'):  #.groupby(["G_model", "FA_case", "G_iteration"]):
        df, df_styled, df_latex, df_config_name = generate_latex_table(df, index, suffix="counterfactual_general")
        print(f"\n\n==================\n" + df_config_name + f"\n==================\n\n {df}", file=file, flush=True)
        print(df_config_name)
        save_table(df_latex, df_config_name)
        display(df_styled)
        # break


# %%
