import io
import os
from typing import Any, Callable
import numpy as np
from thesis_commons.functions import stack_data
from thesis_viability.sparcity.sparcity_metric import SparcityMetric
from thesis_viability.feasibility.feasibility_metric import FeasibilityMetric
from thesis_viability.likelihood.likelihood_improvement import ImprovementCalculator
from thesis_viability.helper.base_distances import odds_ratio as dist
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.libcuts import layers, K, losses
import thesis_commons.metric as metric
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
import tensorflow as tf
import pandas as pd
import glob

DEBUG = True

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = Reader(mode=task_mode).init_meta()
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL_SEP)

    fa_batch, fa_seq_len, fa_ft_size = fa_features.shape
    cf_batch, cf_seq_len, cf_ft_size = cf_features.shape

    a = np.repeat(fa_events, cf_batch, axis=0), np.repeat(fa_features, cf_batch, axis=0)
    b = np.repeat(cf_events[None], fa_batch, axis=0).reshape(-1, cf_seq_len), np.repeat(cf_features[None], fa_batch, axis=0).reshape(-1, cf_seq_len, cf_ft_size)

    sparcity_computer = SparcityMetric(reader.vocab_len, reader.max_len)
    sparcity_values = sparcity_computer.compute_valuation(a, b).reshape(fa_batch, cf_batch)
    print(sparcity_values)

    feasibility_computer = FeasibilityMetric(tr_events, tr_features)
    feasibility_values = feasibility_computer.compute_valuation(cf_events, cf_features)
    print(feasibility_values)

    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects)
    improvement_computer = ImprovementCalculator(model, dist)
    improvement_values = improvement_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features)
    print(improvement_values)
