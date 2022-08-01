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
    destination = PATH_PAPER_TABLES / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table.replace("_", "-"))


# %% ------------------------------
def decode_results(reader, events, features, labels):
    combined = np.concatenate(
        [
            features,
            events[..., None],
            np.repeat(labels[..., None], events.shape[1], axis=1),
            np.repeat(np.arange(len(events))[..., None, None], events.shape[1], axis=1),
        ],
        axis=-1,
    )
    combined

    df_reconstructed = pd.DataFrame(combined.reshape((-1, combined.shape[-1])))
    df_reconstructed = df_reconstructed.rename(columns={ft.index: ft.col for ft in reader.feature_info.all_cols})

    finfo = reader.feature_info
    mapping = reader.pipeline.mapping
    preprocessors = reader.pipeline.collect_as_dict()

    df_postprocessed = df_reconstructed.copy()
    for var_type, columns in mapping.items():
        # for primary_var, sub_var in columns.items():
        #     if sub_var == False:
        #         continue
        if (var_type == CDType.CAT) and (len(columns)):
            ppr = preprocessors.get(CDType.CAT)
            df_postprocessed = ppr.backward(data=df_postprocessed)
        if (var_type == CDType.NUM) and (len(columns)):
            ppr = preprocessors.get(CDType.NUM)
            df_postprocessed = ppr.backward(data=df_postprocessed)
        if (var_type == CDType.BIN) and (len(columns)):
            ppr = preprocessors.get(CDType.BIN)
            df_postprocessed = ppr.backward(data=df_postprocessed)

    df_postprocessed[reader.col_activity_id] = df_postprocessed[reader.col_activity_id].transform(lambda x: reader.idx2vocab[x])
    df_postprocessed[df_postprocessed[reader.col_activity_id] == reader.pad_token] = None
    df_postprocessed[df_postprocessed[reader.col_activity_id] == reader.start_token] = None
    df_postprocessed[df_postprocessed[reader.col_activity_id] == reader.end_token] = None
    return df_postprocessed


decode_results(reader, events, features, labels)
# %%

dict_with_cases = pickle.load(io.open(PATH_RESULTS_MODELS_OVERALL / "dice4el" / 'results.pkl', 'rb'))
dict_with_cases


# %%
def convert_to_dice4el_format(reader, df_post_fa, prefix=""):
    convert_to_dice4el_format = df_post_fa.groupby("id").apply(
        lambda x: {
            prefix + '_' + 'amount': list(x["AMOUNT_REQ"]),
            prefix + '_' + 'activity': list(x[reader.col_activity_id]),
            prefix + '_' + 'resource': list(x["Resource"]),
            # prefix + '_' + 'feasibility': list(x.feasibility)[0],
            prefix + '_' + 'label': list(x.label),
            prefix + '_' + 'id': list(x.id)[0]
        }).to_dict()
    sml = pd.DataFrame(convert_to_dice4el_format).T.reset_index(drop=True)
    return sml


factuals = dict_with_cases.get('_factuals')
events, features, llh, viability = factuals.all
df_post_fa = decode_results(reader, events, features, llh > 0.5)
# feasibility = [1]*len(events[0])
# df_post_fa["feasibility"] = feasibility
# df_post_fa["type"] = "fa"

# display(convert_to_dice4el_format(reader, df_post_fa))
rapper_name = 'CBG_CBGW_IM'


def zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name):
    collector = []
    for idx, (factual, counterfactuals) in enumerate(zip(factuals, dict_with_cases.get(rapper_name))):
        events, features, llh, viability = factual.all
        df_post_fa = decode_results(reader, events, features, llh > 0.5)
        # df_post_fa["feasibility"] = viability.dllh if viability else 0
        df_post_fa["id"] = idx
        fa_line = convert_to_dice4el_format(reader, df_post_fa, "fa")
        for cf_id in range(len(counterfactuals)):
            events, features, llh, viability = counterfactuals[cf_id:cf_id + 1].all
            df_post_cf = decode_results(reader, events, features, llh > 0.5)
            # feasibility = viability.dllh
            # feasibility = viability.sparcity
            # feasibility = viability.similarity
            # feasibility = viability.delta
            df_post_cf["id"] = cf_id
            cf_line = convert_to_dice4el_format(reader, df_post_cf, "cf")
            cf_line["feasibility"] = viability.dllh[0][0]
            cf_line["sparcity"] = viability.sparcity[0][0]
            cf_line["similarity"] = viability.similarity[0][0]
            cf_line["delta"] = viability.ollh[0][0]
            cf_line["viability"] = viability.viabs[0][0]

            merged = pd.concat([fa_line, cf_line], axis=1)
            collector.append(merged)

    all_results = pd.concat(collector).sort_values(["feasibility", "viability"], ascending=True)
    return all_results


all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
all_results


# %%
def expand_again(all_results):
    cols = {
        0: "fa_activity",
        1: "fa_amount",
        2: "fa_resource",
        3: "cf_activity",
        4: "cf_amount",
        5: "cf_resource",
    }
    df_collector = []
    for idx, row in tqdm(all_results.iterrows(), total=len(all_results)):
        tmp_df = pd.DataFrame([
            row["fa_activity"],
            row["fa_amount"],
            row["fa_resource"],
            row["cf_activity"],
            row["cf_amount"],
            row["cf_resource"],
        ]).T
        # tmp_df["fa_amount"] = row["fa_feasibility"]
        tmp_df["fa_label"] = row["fa_label"]
        # tmp_df["cf_amount"] = row["cf_feasibility"]
        tmp_df["cf_label"] = row["cf_label"]
        tmp_df["fa_id"] = row["fa_id"]
        tmp_df["cf_id"] = row["cf_id"]
        tmp_df["cf_feasibility"] = row["feasibility"]
        tmp_df["cf_sparcity"] = row["sparcity"]
        tmp_df["cf_similarity"] = row["similarity"]
        tmp_df["cf_delta"] = row["delta"]
        tmp_df["cf_viability"] = row["viability"]
        df_collector.append(pd.DataFrame(tmp_df))
    new_df = pd.concat(df_collector).rename(columns=cols)
    new_df["cf_resource"] = new_df["cf_resource"]  #.astype(str)
    new_df["fa_resource"] = new_df["fa_resource"]  #.astype(str)
    new_df = new_df.infer_objects()
    return new_df


new_df = expand_again(all_results)
new_df
# %%

# %%
iterator = iter(new_df.groupby(["fa_id", "cf_id"]))
# %%
df_styled = next(iterator)[1]
df_styled
# %% ------------------------------

# %%
index = -2
C_SEQ = "Sequence"
C_FA = f"Factual {C_SEQ}"
C_CF = f"Counterfactual {C_SEQ}"


def generate_latex_table(all_results, index, suffix="", caption=""):
    cols = {
        'fa_activity': (C_FA, "Activity"),
        'fa_amount': (C_FA, "Amount"),
        'fa_resource': (C_FA, "Resource"),
        'fa_label': (C_FA, 'Outcome'),
        'cf_activity': (C_CF, "Activity"),
        'cf_amount': (C_CF, "Amount"),
        'cf_resource': (C_CF, "Resource"),
        'cf_label': (C_CF, 'Outcome'),
    }
    df = expand_again(all_results.iloc[index:index + 1]).rename(columns=cols).iloc[:, :-7]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.loc[:, [C_FA, C_CF]]
    # something.iloc[:, [1,4]] = something.iloc[:, [1,4]].astype(int)
    # something = something.dropna(axis=0)
    df = df[df.notnull().any(axis=1)]
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace("_", "-").str.replace("None", "")
    df.iloc[:, 4] = df.iloc[:, 4].astype(str).str.replace("_", "-").str.replace("None", "")
    df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace(".0", "").str.replace("None", "")
    df.iloc[:, 6] = df.iloc[:, 6].astype(str).str.replace(".0", "").str.replace("None", "")
    # df = df[~(df[(C_FA, "Resource")]=="nan")]

    df_styled = df.style.format(
        # escape='latex',
        precision=0,
        na_rep='',
        thousands=" ",
    ).hide(None)

    df_latex = df_styled.to_latex(
        multicol_align='l',
        # column_format='l',
        caption=f"Shows a factual and the corresponding counterfactual generated. {caption}",
        label=f"tbl:example-cf-{suffix}",
        hrules=True,
    )
    return df, df_styled, df_latex

# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM'
caption = "This counterfactual was generated by the evolutionary algorithm. It is the result which appears to have the highest viability score."
all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 1, "evo", caption)
save_table(df_latex, "example_cf1")
display(df_latex)
display(df_styled)

# %%
# rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_RPR_IM'
# all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
# df, df_styled, df_latex = generate_latex_table(all_results[all_results["feasibility"].between(0,1, 'neither')].groupby("fa_id").tail(1), 0, "evo", caption)
# # save_table(df_latex, "example_cf1")
# display(df_latex)
# display(df_styled)
# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_RR_IM'
all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 0, "evo", caption)
# save_table(df_latex, "example_cf1")
display(df_latex)
display(df_styled)

# %%
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM' # feasibility filter
caption = "This counterfactual has a non-zero feasibility and has the highest viability among the results generated by the evolutionary algorithm."
all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results[all_results["feasibility"]>0].groupby("fa_id").tail(1), 0, "evo_feasibility", caption)
save_table(df_latex, "example_cf2")
display(df_latex)
display(df_styled)
# %%
rapper_name = 'CBG_CBGW_IM'
caption = "This counterfactuals was generated by the case-based model. The counterfactual seems far more viable than the one generated by the evolutionary algorithm."
all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
df, df_styled, df_latex = generate_latex_table(all_results.groupby("fa_id").tail(1), 0, "cbg", caption)
save_table(df_latex, "example_cf3")
display(df_latex)
display(df_styled)

# %%
import textdistance

rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM'
caption = "This counterfactual was generated by the evolutionary algorithm. It is the result which appears to have the highest viability score."
all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
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
for rapper_name in dict_with_cases.keys():
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
    all_results = zip_fa_with_cf(reader, dict_with_cases, factuals, rapper_name)
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
stats_df_for_paper = stats_df.pivot(values=["Value"], index=["Model", "Dimension"], columns=["Property"]).drop(["ES_EGW_SBI_ES_OPC_SBM_RPR_IM", "ES_EGW_SBI_ES_OPC_SBM_RR_IM"], axis=0, error='ignore').droplevel(0, axis=1)#.droplevel("Property")
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
