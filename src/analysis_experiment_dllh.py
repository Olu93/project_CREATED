# %%
import datetime
import io
import os
import pathlib
import sys
import traceback
import itertools as it
from typing import Union
import tensorflow as tf
from thesis_readers.readers.OutcomeReader import OutcomeDice4ELEvalReader, OutcomeDice4ELReader

keras = tf.keras
from tqdm import tqdm
import time
from thesis_commons.constants import (PATH_PAPER_FIGURES, PATH_PAPER_TABLES, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, CDType)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc
import seaborn as sns
from IPython.display import display
# pd.set_option('display.max_columns', 15)
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
experiment_name = "dllh"
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
    destination = PATH_PAPER_TABLES / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table.replace("_", "-"))


# %%
reader.decode_results(events, features, labels)
# %%

list_of_cfs = pickle.load(io.open(PATH_RESULTS_MODELS_OVERALL / experiment_name/ 'results.pkl', 'rb'))
list_of_cfs


# %%
# def convert_to_dice4el_format(reader, df_post_fa, prefix=""):
#     convert_to_dice4el_format = df_post_fa.groupby("id").apply(
#         lambda x: {
#             prefix + '_' + 'amount': list(x["AMOUNT_REQ"]),
#             prefix + '_' + 'activity': list(x[reader.col_activity_id]),
#             prefix + '_' + 'resource': list(x["Resource"]),
#             # prefix + '_' + 'feasibility': list(x.feasibility)[0],
#             prefix + '_' + 'label': list(x.label),
#             prefix + '_' + 'id': list(x.id)[0]
#         }).to_dict()
#     sml = pd.DataFrame(convert_to_dice4el_format).T.reset_index(drop=True)
#     return sml



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
    cf_df_tmp = reader.decode_results( events, features, llh > 0.5, sparcity, similarity, feasibility, delta, viability)
    cf_df_tmp.columns = [f"CF_{col}" for col in cf_df_tmp.columns]
    events, features = fa.cases
    llh = fa.likelihoods
    fa_df_tmp = reader.decode_results( events, features, llh > 0.5)
    fa_df_tmp.columns = [f"FA_{col}" for col in fa_df_tmp.columns]
    df_tmp = pd.concat([cf_df_tmp, fa_df_tmp], axis=1)
    df_tmp["G_model"] = name
    df_tmp["G_iteration"] = iteration
    df_tmp["G_model_num"] = model_num
    collector.append(df_tmp)
all_results = pd.concat(collector)    

# all_results["rank"] = all_results.groupby(["model", "result_id"]).apply(lambda df: list(range(len(df))))
# all_results = reader.zip_fa_with_cf(dict_with_cases, factuals, rapper_name)
all_results["G_step"]= all_results.groupby(["G_model", "FA_case", "G_iteration"]).cumcount()
all_results
# %%
# %%


def generate_latex_table(counterfactual, index, suffix="", caption=""):
    C_SEQ = "Sequence"
    C_FA = f"Factual {C_SEQ}"
    C_CF = f"Counterfactual {C_SEQ}"
    cols = {
        'FA_concept:name': (C_FA, "Activity"),
        'FA_AMOUNT_REQ': (C_FA, "Amount"),
        'FA_Resource': (C_FA, "Resource"),
        'FA_label': (C_FA, 'Outcome'),
        'CF_concept:name': (C_CF, "Activity"),
        'CF_AMOUNT_REQ': (C_CF, "Amount"),
        'CF_Resource': (C_CF, "Resource"),
        'CF_label': (C_CF, 'Outcome'),
        'CF_sparcity': (C_CF, 'Sparcity'),
        'CF_similarity': (C_CF, 'Similarity'),
        'CF_feasibility': (C_CF, 'Feasibility'),
        'CF_delta': (C_CF, 'Delta'),
        'CF_viability': (C_CF, 'Viability'),
    }
    
    df = counterfactual.rename(columns=cols)[list(cols.values())]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.loc[:, [C_FA, C_CF]]
    # something.iloc[:, [1,4]] = something.iloc[:, [1,4]].astype(int)
    # something = something.dropna(axis=0)
    df = df[df.notnull().any(axis=1)]
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace("_", "-", regex=False).str.replace("None", "", regex=False)
    df.iloc[:, 4] = df.iloc[:, 4].astype(str).str.replace("_", "-", regex=False).str.replace("None", "", regex=False)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace("<", "-", regex=False).str.replace(">", "-", regex=False)
    df.iloc[:, 4] = df.iloc[:, 4].astype(str).str.replace("<", "-", regex=False).str.replace(">", "-", regex=False)    
    df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace(".0", "", regex=False).str.replace("None", "", regex=False)
    df.iloc[:, 6] = df.iloc[:, 6].astype(str).str.replace(".0", "", regex=False).str.replace("None", "", regex=False)
    # df = df[~(df[(C_FA, "Resource")]=="nan")]

    df_styled = df.style.format(
        # escape='latex',
        precision=0,
        na_rep='',
        thousands=" ",
    ).hide(None)
    config_name = "-".join([str(e) for e in index]).replace('_', '-')
    df_latex = df_styled.to_latex(
        multicol_align='l',
        # column_format='l',
        caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
        label=f"tbl:example-cf-{'-'.join(config_name)}",
        hrules=True,
    )
    return df, df_styled, df_latex, config_name

# all_results.groupby(["G_model_num", "G_step", "G_iteration"]).tail(1)
file = io.open("dllh_saved_results_final.txt", "w")
for index, df in all_results.groupby(["G_model", "G_step", "G_iteration"]).tail(1).groupby(["G_model", "G_iteration"]):#.groupby(["G_model", "FA_case", "G_iteration"]):
    df, df_styled, df_latex, df_config_name = generate_latex_table(df, list(index))
    print(f"\n\n==================\n"+df_config_name+f"\n==================\n\n {df}", file=file, flush=True)
    # print(df, file=file)
# %%
rapper_name = 'ES_EGW_CBI_ES_OPC_SBM_FSR_IM'
caption = "This counterfactual was generated by the evolutionary algorithm. It is the result which appears to have the highest viability score."
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 0, "evo", caption)
# save_table(df_latex, "example_cf1")
display(df_latex)

# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM'
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 0, "evo", caption)
# save_table(df_latex, "example_cf1")
display(df_latex)
display(df_styled)
# %%
rapper_name = 'ES_EGW_CBI_ES_OPC_SBM_RR_IM'
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 1, "evo", caption)
# save_table(df_latex, "example_cf1")
display(df_latex)
display(df_styled)
# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_RR_IM'
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 1, "evo", caption)
# save_table(df_latex, "example_cf1")
display(df_latex)
display(df_styled)

# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM' # feasibility filter
caption = "This counterfactual has a non-zero feasibility and has the highest viability among the results generated by the evolutionary algorithm."
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results[all_results["feasibility"]>0].groupby("fa_id").tail(1), 0, "evo_feasibility", caption)
save_table(df_latex, "example_cf2")
display(df_latex)
display(df_styled)
# %%
rapper_name = 'CBG_CBGW_IM'
caption = "This counterfactuals was generated by the case-based model. The counterfactual seems far more viable than the one generated by the evolutionary algorithm."
all_results = reader.zip_fa_with_cf(list_of_cfs, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 0, "cbg", caption)
save_table(df_latex, "example_cf3")
display(df_latex)
display(df_styled)

# %%
import textdistance

rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM'
caption = "This counterfactual was generated by the evolutionary algorithm. It is the result which appears to have the highest viability score."
all_results = zip_fa_with_cf(reader, list_of_cfs, factuals, rapper_name)
all_results

# %%
import textdistance


def pad_length(x, to_length, padding_value=None):
    return x + ([padding_value] * (to_length - len(x)))


def get_L2(x, y):
    max_length = max(len(x), len(y))
    padded_x = pad_length(x, max_length)
    padded_y = pad_length(y, max_length)
    assert (len(padded_x) == len(padded_y))

    distance = sum([0 if padded_x[idx] == padded_y[idx] else 1 for idx in range(len(padded_x))])

    return distance**(1 / 2)


def strip_none(sequence):
    return [r for r in sequence if r]


def create_stat(model, dim, property, value):
    return {
        "Model": model,
        "Dimension": dim,
        "Property": property,
        "Value": value,
    }


# all_results.apply(lambda row: print(row["fa_activity"]))

model_stats_dice4el = []
all_possible_activities = reader.original_data.groupby(reader.col_case_id)[reader.col_activity_id].apply(tuple)
all_possible_resources = reader.original_data.groupby(reader.col_case_id)["Resource"].apply(tuple)
for rapper_name in list_of_cfs.keys():
    display(f"--------------------")
    display(f"{rapper_name}:")
    display(f"--------------------")
    if rapper_name == "_factuals":
        continue
    collector_l2_activity = []
    collector_l2_resource = []
    collector_sparcity_activity = []
    collector_sparcity_resource = []
    collector_diversity_activity = []
    collector_diversity_resource = []
    all_results = zip_fa_with_cf(reader, list_of_cfs, factuals, rapper_name)
    cf_activities = all_results["cf_activity"].apply(strip_none).apply(tuple)
    cf_resources = all_results["cf_resource"].apply(strip_none).apply(tuple)
    cf_compare_all_activities = (cf_activities.values[:, None] != cf_activities.values[None])
    cf_compare_all_resources = (cf_resources.values[:, None] != cf_resources.values[None])

    for idx, row in all_results.iterrows():
        collector_l2_activity.append(get_L2(strip_none(row["fa_activity"]), strip_none(row["cf_activity"])))
        collector_l2_resource.append(get_L2(strip_none(row["fa_resource"]), strip_none(row["cf_resource"])))
        collector_sparcity_activity.append(textdistance.levenshtein.distance(strip_none(row["fa_activity"]), strip_none(row["cf_activity"])))
        collector_sparcity_resource.append(textdistance.levenshtein.distance(strip_none(row["fa_resource"]), strip_none(row["cf_resource"])))

    model_stats_dice4el.append(create_stat(rapper_name, "Activity", "Proximity", np.mean(np.array(collector_l2_activity)**2)))
    model_stats_dice4el.append(create_stat(rapper_name, "Activity", "Sparsity", np.mean(collector_sparcity_activity)))
    model_stats_dice4el.append(create_stat(rapper_name, "Activity", "Diversity", np.mean(cf_compare_all_activities)))
    model_stats_dice4el.append(create_stat(rapper_name, "Activity", "Plausibility", np.mean(cf_activities.isin(all_possible_activities))))
    model_stats_dice4el.append(create_stat(rapper_name, "Resource", "Proximity", np.mean(np.array(collector_l2_resource)**2)))
    model_stats_dice4el.append(create_stat(rapper_name, "Resource", "Sparsity", np.mean(collector_sparcity_resource)))
    model_stats_dice4el.append(create_stat(rapper_name, "Resource", "Diversity", np.mean(cf_compare_all_resources)))
    model_stats_dice4el.append(create_stat(rapper_name, "Resource", "Plausibility", np.mean(cf_resources.isin(all_possible_resources))))

    # model_stats_dice4el.append({
    #     "Model": rapper_name,
    #     "Dimension": "Activity",
    #     "Proximity": np.mean(np.array(collector_l2_activity)**2),
    #     "Sparsity": np.mean(collector_sparcity_activity),
    #     "Deiversity": np.mean(cf_compare_all_activities),
    #     "Plausibility": cf_activities.isin(all_possible_activities).mean(),
    # })
    # model_stats_dice4el.append({
    #     "Model": rapper_name,
    #     "Dimension": "Activity",
    #     "Proximity": np.mean(np.array(collector_l2_activity)**2),
    #     "Sparsity": np.mean(collector_sparcity_activity),
    #     "Deiversity": np.mean(cf_compare_all_activities),
    #     "Plausibility": cf_activities.isin(all_possible_activities).mean(),
    # })
    # "Proximity": np.mean(((np.array(collector_l2_activity)**2) + (np.array(collector_l2_resource)**2))**(1 / 2)),
    # "Sparsity": np.mean(np.array(collector_sparcity_activity) + np.array(collector_sparcity_resource)),
    # 'Diversity': np.mean(cf_compare_all_activities.mean(-1) + cf_compare_all_resources.mean(-1)),
    # 'Plausibility': np.mean(cf_activities.isin(all_possible_activities).mean() + cf_resources.isin(all_possible_resources).mean()),
stats_df = pd.DataFrame(model_stats_dice4el).replace({"CBG_CBGW_IM": "Casebased Generator", "RG_RGW_IM": "Random Generator", "ES_EGW_SBI_ES_OPC_SBM_FSR_IM": "Evoluationary: SBI_ES_OPC_SBM_FSR"})
stats_df

# %%

# stats_df = stats_df.set_index("Model")
stats_df_for_paper = stats_df.pivot(values=["Value"], index=["Model", "Dimension"], columns=["Property"]).drop(["ES_EGW_SBI_ES_OPC_SBM_RPR_IM", "ES_EGW_SBI_ES_OPC_SBM_RR_IM"], axis=0, errors='ignore').droplevel(0, axis=1)#.droplevel("Property")
stats_df_for_paper
# %%
df_styled = stats_df_for_paper.style.format(
        # escape='latex',
        # precision=0,
        na_rep='',
        thousands=" ",
    )
df_latex = df_styled.to_latex(
        multicol_align='l',
        # column_format='l',
        caption=f"Shows the mean result of each models' result with respect to diversity, plausibility proximity and sparsity.",
        label=f"tbl:exp6",
        hrules=True,
    )
save_table(df_latex, "exp6-tbl")
display(df_latex)
display(df_styled)
# %%
# %%
# %%
