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
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
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

# %% ------------------------------
(events, features), label = reader.get_dataset_example(ds_mode=DatasetModes.ALL, ft_mode=FeatureModes.FULL)
reader.decode_matrix_str(events)

    
# %%
reader.feature_info.
# %%
