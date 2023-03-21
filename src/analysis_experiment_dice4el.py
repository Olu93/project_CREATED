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
from thesis_readers.readers.OutcomeReader import OutcomeDice4ELEvalReader, OutcomeDice4ELReader
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

ds_name = "OutcomeDice4ELReader"
reader: OutcomeDice4ELReader = OutcomeDice4ELReader.load(PATH_READERS / ds_name)
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

list_of_cfs = pickle.load(io.open(PATH_RESULTS_MODELS_OVERALL / experiment_name / 'results.pkl', 'rb'))
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
C_D4 = 'D4EL'
COL_MAPPER = {C_FA: "Factual Seq.", C_CF: "Our CF Seq.", C_D4: "DiCE4EL CF Seq."}

C_READER_ACTIVITY = reader.col_activity_id
C_READER_AMOUNT = "AMOUNT_REQ"
C_READER_RESOURCE = "Resource"
C_READER_OUTCOME = reader.col_outcome

C_ACTIVITY = 'Activity'
C_RESOURCE = 'Resource'
C_AMOUNT = 'Amount'
C_OUTCOME = 'Outcome'

C_NAME_MAPPER = {C_READER_ACTIVITY: C_ACTIVITY, C_READER_RESOURCE: C_RESOURCE, C_READER_AMOUNT: C_AMOUNT, C_READER_OUTCOME: C_OUTCOME}
GLUE_GROUPS = [C_FA, C_CF, C_D4]
GLUE_COLS = [G_CASE, G_NAME, G_ITER, G_MODL, G_STEP]
# GLUE_TAIL_GROUPER = list([tuple(els) for els in np.array(GLUE_COLS)[[1, 2, 4]]])
GLUE_TAIL_GROUPER = [G_NAME, G_ITER, G_STEP]
GLUE_GID = [G_NAME, G_CASE, G_ITER]
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
    df_tmp[G_CASE] = cf_df_tmp["case"].astype(int)
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
    df_tmp[G_CASE] = fa_df_tmp["case"].astype(int)
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


collector = []
d4el_df = reader._d4e_cfs.groupby("caseid_seed").head(1)
for model_num, el in enumerate(list_of_cfs):
    name, cf, fa, iteration = el.values()
    d4el_df_tmp = expand_d4el(reader, d4el_df[iteration:iteration + 1])
    d4el_df_tmp[C_READER_ACTIVITY] = d4el_df_tmp[C_READER_ACTIVITY].str.replace("_COMPLETE", "")
    # d4el_df_tmp.columns = pd.MultiIndex.from_product([[C_D4], d4el_df_tmp.columns])
    old_cols = [col for col in d4el_df_tmp.columns if col != 'case']
    new_cols = pd.MultiIndex.from_product([[C_D4], old_cols])
    renamer = dict(zip(old_cols, new_cols))
    # renamer["case"] = G_CASE
    df_tmp = pd.DataFrame(d4el_df_tmp)  #.rename(columns=renamer)
    df_tmp = df_tmp.rename(columns=renamer)[list(renamer.values())]
    df_tmp.columns = pd.MultiIndex.from_tuples(df_tmp.columns)
    df_tmp[G_NAME] = name
    df_tmp[G_CASE] = None
    df_tmp[G_ITER] = iteration
    df_tmp[G_MODL] = model_num
    collector.append(df_tmp)
d4el_all_results = pd.concat(collector)
d4el_all_results[GLUE_COLS[-1]] = d4el_all_results.groupby(GLUE_COLS[1:-1]).cumcount()
d4el_all_results
# %%
df_merged = all_results.copy()
df_merged = df_merged.merge(fa_all_results, how='left', on=GLUE_COLS)
df_merged = df_merged.merge(d4el_all_results, how='left', suffixes=(None, "_2"), on=GLUE_COLS[1:3] + GLUE_COLS[4:])
df_merged["gid"] = df_merged[GLUE_GID].astype(str).apply('_'.join, axis=1)
df_merged

df_merged = df_merged.set_index("gid")
df_merged = df_merged.rename(columns=C_NAME_MAPPER, level=1)

df_merged.loc(axis=1)[:, C_RESOURCE] = df_merged.loc(axis=1)[:, C_RESOURCE].astype(str).apply(lambda x: x.str.replace('.0', '').str.replace('UNKNOWN', 'other')).values
# df_merged.loc(axis=1)[:, C_AMOUNT] = np.floor(df_merged.loc(axis=1)[:, C_AMOUNT]).astype(str).apply(lambda x: x.str.replace('.0', '')).values
df_merged.loc(axis=1)[:, C_OUTCOME] = np.floor(df_merged.loc(axis=1)[:, C_OUTCOME]).astype(str).apply(lambda x: x.str.replace('.0', '')).values
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
with io.open("dice4el_saved_results_d4el.txt", "w") as file:
    best_ones = df_merged.groupby(GLUE_TAIL_GROUPER).tail(1).reset_index()
    for index, df in best_ones.groupby('gid'):  #.groupby(["G_model", "FA_case", "G_iteration"]):
        df, df_styled, df_latex, df_config_name = generate_latex_table(df, index, "d4el")
        print(f"\n\n==================\n" + df_config_name + f"\n==================\n\n {df}", file=file, flush=True)
        print(df_config_name)
        save_table(df_latex, df_config_name)
        display(df_styled)
        # break

# %%

# %%
import textdistance
from typing import List


def get_similarity(x, y):
    max_length = max(len(x), len(y))
    padded_x = pad_length(x, max_length)
    padded_y = pad_length(y, max_length)
    assert (len(padded_x) == len(padded_y))

    distance = sum([1 if padded_x[idx] == padded_y[idx] else 0 for idx in range(max_length)])

    return distance / max_length

# def get_similarity(x, y):
#     max_length = max(len(x), len(y))
#     padded_x = pad_length(x, max_length)
#     padded_y = pad_length(y, max_length)
#     assert (len(padded_x) == len(padded_y))

#     distance = sum([0 if padded_x[idx] == padded_y[idx] else 1 for idx in range(len(padded_x))])

#     return distance**(1 / 2)


def strip_none(sequence: List[str]):
    return [r for r in sequence if not ((r == None) or r.startswith("<") or (r == "") or r == 'nan')]


def create_stat(exp, model, case, grp, iteration, dim, property, value):
    return {
        "Experiment": exp,
        "Generator": model,
        "Case": case,
        "Model": grp,
        "Factual": iteration,
        "Dimension": dim,
        "Property": property,
        "Value": value,
    }


@lru_cache
def check_plausbility(element, vault):
    upper_limit = len(element)
    if upper_limit > 1:
        trimmed = list(set([tuple(e[:upper_limit]) for e in vault]))
        return (element in trimmed)
    return 0


model_stats_dice4el = []
odata = reader.original_data
odata[C_READER_ACTIVITY] = odata[C_READER_ACTIVITY].astype(str)
odata[C_READER_RESOURCE] = odata[C_READER_RESOURCE].astype(str).str.replace(".0", "")
all_possible_activities = list(odata.groupby(reader.col_case_id)[C_READER_ACTIVITY].apply(strip_none).apply(tuple).values)
all_possible_activities = tuple(sorted(set(all_possible_activities)))
all_possible_resources = list(odata.groupby(reader.col_case_id)[C_READER_RESOURCE].apply(strip_none).apply(tuple).values)
all_possible_resources = tuple(sorted(set(all_possible_resources)))
for index, df in tqdm(df_merged.groupby(df_merged.index), total=df_merged.index.nunique()):
    iteration = index.split("_")[-1]
    case = index.split("_")[-2]
    name = "-".join(index.split("_")[:-2])

    factual = df[C_FA]
    factual_activities = tuple(factual[[C_ACTIVITY]].astype(str).apply(strip_none)[C_ACTIVITY])
    factual_resources = tuple(factual[[C_RESOURCE]].astype(str).apply(strip_none)[C_RESOURCE])
    for competitor in GLUE_GROUPS[1:]:
        all_results: pd.DataFrame = df[competitor]
        cf_activities = tuple(all_results[[C_ACTIVITY]].astype(str).apply(strip_none)[C_ACTIVITY].values)
        cf_resources = tuple(all_results[[C_RESOURCE]].astype(str).apply(strip_none)[C_RESOURCE].values)

        model_stats_dice4el.append(
            create_stat(
                index,
                name,
                case,
                competitor,
                iteration,
                C_ACTIVITY,
                "Proximity",
                get_similarity(strip_none(factual_activities), strip_none(cf_activities)),
            ))
        model_stats_dice4el.append(
            create_stat(
                index,
                name,
                case,
                competitor,
                iteration,
                C_ACTIVITY,
                "Sparsity",
                get_similarity(strip_none(factual_resources), strip_none(cf_resources)),
            ))
        # model_stats_dice4el.append(create_stat(index, name, case, competitor, iteration, C_ACTIVITY, "PList", np.array(collector_l2_activity)))
        model_stats_dice4el.append(create_stat(
            index,
            name,
            case,
            competitor,
            iteration,
            C_ACTIVITY,
            "Plausibility",
            check_plausbility(cf_activities, all_possible_activities),
        ))
        model_stats_dice4el.append(
            create_stat(
                index,
                name,
                case,
                competitor,
                iteration,
                C_RESOURCE,
                "Proximity",
                textdistance.levenshtein.normalized_similarity(strip_none(factual_activities), strip_none(cf_activities)),
            ))
        model_stats_dice4el.append(
            create_stat(
                index,
                name,
                case,
                competitor,
                iteration,
                C_RESOURCE,
                "Sparsity",
                textdistance.levenshtein.normalized_similarity(strip_none(factual_resources), strip_none(cf_resources)),
            ))
        # model_stats_dice4el.append(create_stat(index, name, case, competitor, iteration, C_RESOURCE, "PList", np.array(collector_l2_resource)))
        model_stats_dice4el.append(create_stat(
            index,
            name,
            case,
            competitor,
            iteration,
            C_RESOURCE,
            "Plausibility",
            check_plausbility(cf_resources, all_possible_resources),
        ))

# %%
list_groups = list(df_merged.groupby([G_NAME, G_ITER]))
grp_collector = []
for index, df in tqdm(list_groups, total=len(list_groups)):
    for case, grp in df.groupby(G_CASE):
        
        for competitor in GLUE_GROUPS[1:]:
            all_results: pd.DataFrame = grp[competitor]        
            grp_collector.append({
                "model":index[0],
                "factual":index[1],
                "case":case,
                "competitor":competitor,
                "activities": tuple(all_results[[C_ACTIVITY]].astype(str).apply(strip_none)[C_ACTIVITY].values),
                "resources": tuple(all_results[[C_RESOURCE]].astype(str).apply(strip_none)[C_RESOURCE].values),
            })

inline_data = pd.DataFrame(grp_collector)
inline_data

# %% data

inline_tmp = inline_data.copy()

def get_ill(group):
    comparison_activities = group.values[:,None] == group.values[None]
    remove_diag = comparison_activities #- np.eye(comparison_activities.shape[0])
    ind_similarities = (remove_diag.mean(-1))#/remove_diag.shape[-1]
    ind_diversities = (1-ind_similarities)
    return ind_diversities

inline_tmp["Activity"] = inline_tmp.groupby(["model","factual","competitor"])["activities"].transform(get_ill)
inline_tmp["Resource"] = inline_tmp.groupby(["model","factual","competitor"])["resources"].transform(get_ill)
inline_tmp["Generator"] = inline_tmp["model"].str.replace("_", "-")
inline_tmp["Factual"] = inline_tmp["factual"].astype("str")
inline_tmp["Case"] = inline_tmp["case"].astype("str")
inline_tmp["Model"] = inline_tmp["competitor"].astype("str")
almost_there = inline_tmp.groupby(["Model", "Factual", "Generator"]).mean().reset_index().melt(id_vars=["Model", "Factual", "Generator"],value_vars=["Activity", "Resource"], var_name="Dimension", value_name="Value")
almost_there["Property"] = "Diversity"
almost_there

# %%
stats_df = pd.DataFrame(model_stats_dice4el)
# pd.merge(stats_df,inline_data, how="left", left_on=["Model", "Factual", "Case", "Generator"], right_on=["competitor", "factual", "case", "generator"])

stats_df_for_paper = stats_df.groupby([
    "Generator",
    "Factual",
    "Dimension",
    "Property",
    'Model',
]).mean().reset_index().drop("Case", axis=1)
stats_df_for_paper

# %%
final_df = pd.concat([stats_df_for_paper, almost_there[stats_df_for_paper.columns]])
final_df
# %%
pivot_df = final_df.infer_objects().pivot_table(values=["Value"], index=["Generator", "Dimension", "Factual"], columns=["Model", "Property"]).droplevel(0, axis=1)
pivot_df = pivot_df.rename(columns={"CF": "CREATED"}, level=0)
# pivot_df.head(50)
pivot_df#.head(60)
pivot_df
# %%
df_styled = pivot_df.style.format(
    escape='latex',
    # precision=0,
    na_rep='',
    thousands=" ",
)
df_latex = df_styled.to_latex(
    multicol_align='l',
    multirow_align='t',
    clines='skip-last;data',
    # column_format='l',
    # caption=f"Shows the mean result of each models' result with respect to diversity, plausibility proximity and sparsity.",
    # label=f"tbl:exp6",
    hrules=True,
)
save_table(df_latex, "exp6-tbl")
display(df_latex)
display(df_styled)
# %%
# %%
# %%
