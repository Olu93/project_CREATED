import glob
import io
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

import thesis_commons.metric as metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.functions import stack_data
from thesis_commons.libcuts import K, layers, losses
from thesis_commons.modes import (DatasetModes, FeatureModes, GeneratorModes,
                                  TaskModes)
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_readers import MockReader as Reader
from thesis_viability.feasibility.feasibility_metric import FeasibilityMeasure
from thesis_viability.likelihood.likelihood_improvement import \
    OutcomeImprovementMeasureDiffs as ImprovementMeasure
from thesis_viability.similarity.similarity_metric import SimilarityMeasure
from thesis_viability.sparcity.sparcity_metric import SparcityMeasure

DEBUG = True

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = Reader(mode=task_mode).init_meta()
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)

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
