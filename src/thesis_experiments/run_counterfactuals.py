import io
import os
from typing import Any, Callable
import numpy as np
from thesis_commons.statististics import ResultStatistics
from thesis_generators.models.baselines.random_search import RandomGeneratorModel
from thesis_generators.models.baselines.casebased_heuristic import CaseBasedGeneratorModel
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import SimpleEvolutionStrategy
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.representations import Cases
from thesis_generators.generators.baseline_wrappers import CaseBasedGeneratorWrapper, RandomGeneratorWrapper
from thesis_generators.generators.evo_wrappers import SimpleEvoGeneratorWrapper
from thesis_generators.generators.vae_wrappers import SimpleVAEGeneratorWrapper
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_viability.likelihood.likelihood_improvement import SummarizedNextActivityImprovementMeasureOdds as ImprovementMeasure
from thesis_commons.constants import PATH_MODELS_PREDICTORS, PATH_MODELS_GENERATORS, PATH_RESULTS_COUNTERFACTUALS
from thesis_commons.modes import DatasetModes, FeatureModes
from thesis_commons.modes import TaskModes
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel as Generator
import tensorflow as tf
import pandas as pd
import glob
from thesis_predictors.models.lstms.lstm import OutcomeLSTM

DEBUG_USE_QUICK_MODE = True
DEBUG_USE_MOCK = True
DEBUG_SKIP_VAE = False

if DEBUG_USE_MOCK:
    from thesis_readers import OutcomeMockReader as Reader
else:
    from thesis_readers import OutcomeBPIC12Reader as Reader

if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    k_fa = 3
    topk = 5
    outcome_of_interest = 1
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    feature_len = reader.current_feature_len  # TODO: Change to function which takes features and extracts shape

    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}

    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), cf_labels = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)

    fa_events, fa_features, fa_labels = fa_events[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_features[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_labels[
        fa_labels[:, 0] == outcome_of_interest][:k_fa]
    fa_cases = Cases(fa_events, fa_features, fa_labels)
    vault_cases = Cases(cf_events, cf_features, cf_labels)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    evaluator = ViabilityMeasure(vocab_len, max_len, (tr_events, tr_features), predictor)


    # EVO GENERATOR
    evo_generator = SimpleEvolutionStrategy(max_iter=10, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    cbg_generator = CaseBasedGeneratorModel(vault_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    rng_generator = RandomGeneratorModel(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)

    simple_evo_generator = SimpleEvoGeneratorWrapper(predictor=predictor, generator=evo_generator, evaluator=evaluator, topk=topk)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    random_generator = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))

    stats = ResultStatistics()
    
    if not DEBUG_SKIP_VAE:
        # VAE GENERATOR
        # TODO: Think of reversing cfs
        all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
        vae_generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
        print("GENERATOR")
        vae_generator.summary()
        simple_vae_generator = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
        stats.update(model=simple_vae_generator, data=fa_cases)

    stats.update(model=simple_evo_generator, data=fa_cases)
    stats.update(model=case_based_generator, data=fa_cases)
    stats.update(model=random_generator, data=fa_cases)

    print("")
    print(stats)
    print("")
    print(stats.data)
    print("")
    stats.data.to_csv(PATH_RESULTS_COUNTERFACTUALS/"cf_generation_results.csv")

    print("DONE")