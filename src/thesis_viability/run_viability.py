import io
import os
from typing import Any, Callable
import numpy as np
from thesis_commons.functions import reverse_sequence_2
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.functions import stack_data
from thesis_viability.similarity.similarity_metric import SimilarityMeasure
from thesis_viability.sparcity.sparcity_metric import SparcityMeasure
from thesis_viability.feasibility.feasibility_metric import FeasibilityMeasure
from thesis_viability.likelihood.likelihood_improvement import SummarizedNextActivityImprovementMeasureOdds as ImprovementMeasure
from thesis_viability.helper.base_distances import odds_ratio as dist
from thesis_commons.constants import PATH_MODELS_PREDICTORS, PATH_MODELS_GENERATORS
from thesis_commons.libcuts import layers, K, losses
import thesis_commons.metric as metric
from thesis_readers import OutcomeBPIC12Reader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel as Generator
import tensorflow as tf
import pandas as pd
import glob
from thesis_predictors.models.lstms.lstm import OutcomeLSTM

DEBUG = True



if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 50
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}
    
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)
    fa_events, fa_features = fa_events[:1], fa_features[:1]

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()
    
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    generator = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    generator.summary()
    
    cf_events, cf_features = generator.predict([np.repeat(fa_events, 10, axis=0),np.repeat(fa_features, 10, axis=0) ])    
    cf_events, cf_features = reverse_sequence_2(cf_events), reverse_sequence_2(cf_features)
    viability = ViabilityMeasure(reader.vocab_len, reader.max_len, (tr_events, tr_features), predictor)
    
    viability_values = viability.compute_valuation(fa_events, fa_features, cf_events, cf_features)
    print(viability_values)
    print("DONE")