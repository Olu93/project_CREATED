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
from thesis_readers.helper.constants import DATA_FOLDER
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
    destination = PATH_PAPER_TABLES / f"{filename}.tex"
    with destination.open("w") as f:
        f.write(table.replace("_", "-"))


# %%
decoded = reader.decode_results(events, features, labels)
decoded
# %%
pd.read_csv(DATA_FOLDER/"dataset_dice4el"/"cf_all_examples.csv")
# pickle.load(DATA_FOLDER/"thesis_commons")
# %%
