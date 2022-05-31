import io
import os
from typing import Any, Callable
import numpy as np
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import SimpleEvolutionStrategy
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.representations import Cases
from thesis_generators.generators.evo_generator import SimpleEvoGenerator
from thesis_generators.generators.vae_generator import SimpleVAEGenerator
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
from thesis_readers import OutcomeMockReader as Reader
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
    k_fa = 3
    outcome_of_interest = 1
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}
    
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    fa_events, fa_features, fa_labels = fa_events[fa_labels[:, 0]==outcome_of_interest][:k_fa], fa_features[fa_labels[:, 0]==outcome_of_interest][:k_fa], fa_labels[fa_labels[:, 0]==outcome_of_interest][:k_fa]
    fa_cases = Cases(fa_events, fa_features, fa_labels)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()
    
    evaluator = ViabilityMeasure(reader.vocab_len, reader.max_len, (tr_events, tr_features), predictor)
    
    # VAE GENERATOR
    # TODO: Think of reversing cfs
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    vae_generator:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    vae_generator.summary()
    
    # EVO GENERATOR
    evo_generator = SimpleEvolutionStrategy(100)
    
    simple_vae_generator = SimpleVAEGenerator(predictor=predictor, generator=vae_generator, evaluator=evaluator)
    simple_evo_generator = SimpleEvoGenerator(predictor=predictor, generator=evo_generator, evaluator=evaluator)
    
    vae_results = simple_vae_generator.generate(fa_cases)
    evo_results = simple_evo_generator.generate(fa_cases)
    print("DONE")