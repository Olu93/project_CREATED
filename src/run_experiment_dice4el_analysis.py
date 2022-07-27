# %%
import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO
import traceback
import itertools as it
import tensorflow as tf

from thesis_readers.readers.OutcomeReader import OutcomeDice4ELEvalReader

keras = tf.keras
from keras import models
from tqdm import tqdm
import time
from thesis_commons.config import DEBUG_USE_MOCK, READER
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC, CDType)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statistics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_experiments.commons import build_cb_wrapper, build_evo_wrapper, build_rng_wrapper, build_vae_wrapper, run_experiment
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader, OutcomeBPIC12Reader25
from thesis_readers.helper.helper import get_all_data, get_even_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PModel
from thesis_predictors.helper.runner import Runner as PRunner
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
def convert_to_dice4el_format(reader, df_post_fa):
    convert_to_dice4el_format = df_post_fa.groupby(10).apply(lambda x: {
    'amount': list(x.amount),
    'activity': list(x[reader.col_activity_id]),
    'resource': list(x.resource),
    'feasibility': list(x.feasibility),
    'label': list(x.label)[0]
}).to_dict()
    sml = pd.DataFrame(convert_to_dice4el_format).T
    return sml


# factuals = dict_with_cases.get('_factuals')
# events, features, llh, viability = factuals.all
# df_post_fa = decode_results(reader, events, features, llh > 0.5)
# feasibility = [1]*len(events[0])
# df_post_fa["feasibility"] = feasibility
# df_post_fa["type"] = "fa"

# display(convert_to_dice4el_format(reader, df_post_fa))

for factual_id, counterfactuals in enumerate(dict_with_cases.get('EvoGeneratorWrapper')):
    for cf_id in range(len(counterfactuals)):
        events, features, llh, viability = counterfactuals[cf_id:cf_id+1].all
        df_post_cf = decode_results(reader, events, features, llh > 0.5)
        feasibility = viability.dllh
        df_post_cf["type"] = "cf"
        df_post_cf["feasibility"] = feasibility[0][0]
        # df = pd.concat([df_post_fa, df_post_cf], axis=0)
        display(convert_to_dice4el_format(reader, df_post_cf))

# %%

    # %%

convert_to_dice4el_format(reader, df_post_fa)
