import io
import os
from typing import Any, Callable
import numpy as np
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.representations import Cases
from thesis_generators.generators.vae_generator import SimpleVAEGenerator
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_viability.likelihood.likelihood_improvement import SummarizedNextActivityImprovementMeasureOdds as ImprovementMeasure
from thesis_commons.constants import PATH_MODELS_PREDICTORS, PATH_MODELS_GENERATORS
from thesis_readers import OutcomeMockReader as Reader
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel as GModel
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PredictionModel
import tensorflow as tf
import pandas as pd
import glob
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_commons.callbacks import CallbackCollection
from thesis_predictors.helper.runner import Runner as PredictorRunner

DEBUG = True



if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    embed_dim = 12
    ff_dim = 5
    reader: AbstractProcessLogReader = Reader(mode=task_mode).init_log(True).init_meta()
    # generative_reader = GenerativeDataset(reader)
    train_data = reader.get_dataset_generative(20, DatasetModes.TRAIN, FeatureModes.FULL, flipped_target=True)
    val_data = reader.get_dataset_generative(20, DatasetModes.VAL, FeatureModes.FULL, flipped_target=True)

    DEBUG = True
    model = PredictionModel(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
    r1 = PredictorRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)
    
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    fa_events, fa_features, fa_labels = fa_events[fa_labels[:, 0]==1][:1], fa_features[fa_labels[:, 0]==1][:1], fa_labels[fa_labels[:, 0]==1]
    fa_cases = Cases(fa_events, fa_features, fa_labels)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()
    
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    generator:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    generator.summary()
    
    evaluator = ViabilityMeasure(reader.vocab_len, reader.max_len, (tr_events, tr_features), predictor)
    # TODO: Think of reversing cfs
    simple_vae_generator = SimpleVAEGenerator(predictor=predictor, generator=generator, evaluator=evaluator)    
    print("DONE")