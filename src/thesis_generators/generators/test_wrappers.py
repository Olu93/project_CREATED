import glob
import io
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from thesis_commons.constants import (PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS)
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases
from thesis_generators.generators.baseline_wrappers import (
    CaseBasedGeneratorWrapper, RandomGeneratorWrapper)
from thesis_generators.generators.evo_wrappers import SimpleEvoGeneratorWrapper
from thesis_generators.generators.vae_wrappers import SimpleVAEGeneratorWrapper
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGeneratorModel
from thesis_generators.models.baselines.random_search import \
    RandomGeneratorModel
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import \
    SimpleEvolutionStrategy
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import OutcomeMockReader as Reader
from thesis_viability.likelihood.likelihood_improvement import \
    SummarizedNextActivityImprovementMeasureOdds as ImprovementMeasure
from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG = True


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    k_fa = 3
    outcome_of_interest = 1
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    feature_len = reader.current_feature_len  # TODO: Change to function which takes features and extracts shape

    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}

    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)

    fa_events, fa_features, fa_labels = fa_events[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_features[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_labels[
        fa_labels[:, 0] == outcome_of_interest][:k_fa]
    fa_cases = Cases(fa_events, fa_features, fa_labels)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    evaluator = ViabilityMeasure(vocab_len, max_len, (tr_events, tr_features), predictor)

    # VAE GENERATOR
    # TODO: Think of reversing cfs
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    vae_generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    vae_generator.summary()

    # EVO GENERATOR
    evo_generator = SimpleEvolutionStrategy(10, evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)

    # Baselines
    cbg_generator = CaseBasedGeneratorModel((cf_events, cf_features), evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    rng_generator = RandomGeneratorModel(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)

    simple_vae_generator = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, topk=5)
    simple_evo_generator = SimpleEvoGeneratorWrapper(predictor=predictor, generator=evo_generator, evaluator=evaluator, topk=5)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=5)
    random_generator = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, topk=5)

    results = {
        type(simple_vae_generator).__name__:simple_vae_generator.generate(fa_cases),
        type(simple_evo_generator).__name__:simple_evo_generator.generate(fa_cases),
        type(case_based_generator).__name__:case_based_generator.generate(fa_cases),
        type(random_generator).__name__:random_generator.generate(fa_cases),
    }

    print("\nRESULTS\n")
    for key, res in results.items():
        print(f"{key}:\n{res[0].viabilities}\n")

    print("DONE")