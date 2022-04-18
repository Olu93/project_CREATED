import io
import os
from typing import Any, Callable
import numpy as np
from thesis_commons.functions import stack_data
from thesis_viability.similarity.similarity_metric import SimilarityMeasure
from thesis_viability.sparcity.sparcity_metric import SparcityMeasure
from thesis_viability.feasibility.feasibility_metric import FeasibilityMeasure
from thesis_viability.likelihood.likelihood_improvement import ImprovmentMeasureOdds as ImprovementMeasure
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

    sparcity_computer = SparcityMeasure(reader.vocab_len, reader.max_len)
    sparcity_values = sparcity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features)
    print(sparcity_values)

    similarity_computer = SimilarityMeasure(reader.vocab_len, reader.max_len)
    similarity_values = similarity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features)
    print(similarity_values)

    feasibility_computer = FeasibilityMeasure(tr_events, tr_features)
    feasibility_values = feasibility_computer.compute_valuation(cf_events, cf_features)
    print(feasibility_values)

    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects)
    improvement_computer = ImprovementMeasure(model)
    improvement_values = improvement_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features)
    print(improvement_values)
    print("DONE")
