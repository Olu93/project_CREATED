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
from thesis_readers.readers.OutcomeReader import OutcomeDice4ELEvalReader

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

ds_name = "OutcomeDice4ELEvalReader"
reader: OutcomeDice4ELEvalReader = OutcomeDice4ELEvalReader.load(PATH_READERS / ds_name)
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

dict_with_cases = pickle.load(io.open(PATH_RESULTS_MODELS_OVERALL / 'results.pkl', 'rb'))
dict_with_cases


# %%
def convert_to_dice4el_format(reader, df_post_fa, prefix=""):
    convert_to_dice4el_format = df_post_fa.groupby("id").apply(
        lambda x: {
            prefix + '_' + 'amount': list(x.amount),
            prefix + '_' + 'activity': list(x[reader.col_activity_id]),
            prefix + '_' + 'resource': list(x.resource),
            # prefix + '_' + 'feasibility': list(x.feasibility)[0],
            prefix + '_' + 'label': list(x.label)[0],
            prefix + '_' + 'id': list(x.id)[0]
        }).to_dict()
    sml = pd.DataFrame(convert_to_dice4el_format).T
    return sml


factuals = dict_with_cases.get('_factuals')
events, features, llh, viability = factuals.all
df_post_fa = decode_results(reader, events, features, llh > 0.5)
# feasibility = [1]*len(events[0])
# df_post_fa["feasibility"] = feasibility
# df_post_fa["type"] = "fa"

# display(convert_to_dice4el_format(reader, df_post_fa))
# rapper_name = 'CaseBasedGeneratorWrapper'
rapper_name = 'ES_EGW_SBI_ES_OPC_SBM_FSR_IM'
collector = []
for idx, (factual, counterfactuals) in enumerate(zip(factuals, dict_with_cases.get(rapper_name))):
    events, features, llh, viability = factual.all
    df_post_fa = decode_results(reader, events, features, llh > 0.5)
    df_post_fa["feasibility"] = viability.dllh if viability else 0
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
        cf_line["dllh"] = viability.dllh[0][0]
        cf_line["sparcity"] = viability.sparcity[0][0]
        cf_line["similarity"] = viability.similarity[0][0]
        cf_line["delta"] = viability.ollh[0][0]
        cf_line["viability"] = viability.viabs[0][0]

        merged = pd.concat([fa_line, cf_line], axis=1)
        collector.append(merged)

all_results = pd.concat(collector)
all_results


# %%
def expand_again(all_results):
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
        tmp_df["cf_dllh"] = row["dllh"]
        tmp_df["cf_sparcity"] = row["sparcity"]
        tmp_df["cf_similarity"] = row["similarity"]
        tmp_df["cf_delta"] = row["delta"]
        tmp_df["cf_viability"] = row["viability"]
        df_collector.append(pd.DataFrame(tmp_df))
    new_df = pd.concat(df_collector).infer_objects()
    return new_df


new_df = expand_again(all_results)
new_df
# %%
cols = {
    0: "fa_activity",
    1: "fa_amount",
    2: "fa_resource",
    3: "cf_activity",
    4: "cf_amount",
    5: "cf_resource",
}
# new_df = pd.concat(df_collector)
new_df = new_df.rename(columns=cols)
new_df
# %%
iterator = iter(new_df.groupby(["fa_id", "cf_id"]))
# %%
something = next(iterator)[1]
something
# %% ------------------------------

expand_again(all_results.groupby(["fa_id"]).head(1)).rename(columns=cols)
# %%
expand_again(all_results).tail(50).rename(columns=cols)
# %%
something = next(iterator)[1]
save_table(something, "example_cf")
# %% ------------------------------
expand_again(all_results.iloc[-1:]).rename(columns=cols)
# %%
index = -2
def generate_latex_table(all_results, index):
    cols = {
    0: ("Factual", "Activity"),
    1: ("Factual", "Amount"),
    2: ("Factual", "Resource"),
    'fa_label': ("Factual", 'Outcome'),
    3: ("Counterfactual", "Activity"),
    4: ("Counterfactual", "Amount"),
    5: ("Counterfactual", "Resource"),
    'cf_label': ("Counterfactual", 'Outcome'),
}
    something = expand_again(all_results.iloc[index:index+1]).rename(columns=cols).iloc[:, :-7]
    something.columns = pd.MultiIndex.from_tuples(something.columns)
    something = something.loc[:, ["Factual", "Counterfactual"]]
    something.iloc[:, 0] = something.iloc[:, 0].str.replace("_COMPLETE", "").str.replace("_", "-")
    something.iloc[:, 4] = something.iloc[:, 4].str.replace("_COMPLETE", "").str.replace("_", "-")
# something.iloc[:, [1,4]] = something.iloc[:, [1,4]].astype(int)
    something_txt = something.style.format(
    # escape='latex',
    precision=0,
    na_rep='',
    thousands=" ",
).hide(None).to_latex(
    multicol_align='l',
    # column_format='l',
    caption="Shows a factual and the corresponding counterfactual generated.",
    label="tbl:example-cf",
    hrules=True,
)
    
    return something,something_txt
# %%
something, something_txt = generate_latex_table(all_results.groupby("fa_id").tail(1), 0)
save_table(something_txt, "example_cf1")

display(something_txt)
display(something)
# %%
something, something_txt = generate_latex_table(all_results.groupby("fa_id").tail(1), 1)
# save_table(something_txt, "example_cf")
save_table(something_txt, "example_cf2")

display(something_txt)
display(something)
# %%
something, something_txt = generate_latex_table(all_results.groupby("fa_id").tail(1), 2)
# save_table(something_txt, "example_cf")
save_table(something_txt, "example_cf3")

display(something_txt)
display(something)
# %%
something, something_txt = generate_latex_table(all_results.groupby("fa_id").tail(1), 3)
# save_table(something_txt, "example_cf")
save_table(something_txt, "example_cf4")

display(something_txt)
display(something)
# %%
