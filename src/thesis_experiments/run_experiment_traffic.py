import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO
import traceback
import itertools as it
import tensorflow as tf
from thesis_generators.generators.evo_wrappers import EvoGeneratorWrapper

from thesis_readers.readers.OutcomeReader import OutcomeBPIC12Reader50, OutcomeDice4ELEvalReader, OutcomeDice4ELReader, OutcomeTrafficFineReader, OutcomeTrafficShortReader
import numpy as np
keras = tf.keras
from keras import models
from tqdm import tqdm
import time
from thesis_commons.config import DEBUG_USE_MOCK, MAX_ITER_STAGE_1, MAX_ITER_STAGE_3, MUTATION_RATE_STAGE_1, MUTATION_RATE_STAGE_3, READER
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import DataDistribution, DistributionConfig, EmissionProbIndependentFeatures, MarkovChainProbability
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
from thesis_readers.helper.helper import get_all_data, get_d4el_data, get_even_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PModel
from thesis_predictors.helper.runner import Runner as PRunner
import pickle

DEBUG_QUICK_MODE = 0
DEBUG_SKIP_VAE = 1
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 0
DEBUG_SKIP_RNG = 0
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True
CONFIG_IS_FRESH = False


def create_combinations(erate: float, default_mrate: MutationRate, evaluator: ViabilityMeasure):
    all_evo_configs = []
    all_evo_configs.append(
        evolutionary_operations.EvoConfigurator(
            evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
            evolutionary_operations.ElitismSelector(),
            evolutionary_operations.UniformCrosser().set_crossover_rate(0.3) ,
            evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
            evolutionary_operations.RankedRecombiner(),
        ))
    all_evo_configs.append(
        evolutionary_operations.EvoConfigurator(
            evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
            evolutionary_operations.RouletteWheelSelector(),
            evolutionary_operations.OnePointCrosser(),
            evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
            evolutionary_operations.FittestSurvivorRecombiner(),
        ))
    # all_evo_configs.append(
    #     evolutionary_operations.EvoConfigurator(
    #         evolutionary_operations.SamplingBasedInitiator().set_data_distribution(evaluator.data_distribution),
    #         evolutionary_operations.RouletteWheelSelector(),
    #         evolutionary_operations.OnePointCrosser(),
    #         evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
    #         evolutionary_operations.BestBreedRecombiner(),
    #     ))
    return all_evo_configs


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    num_iterations = 5 if DEBUG_QUICK_MODE else MAX_ITER_STAGE_3
    k_fa = 2
    top_k = 10 if DEBUG_QUICK_MODE else 50
    epochs = 5
    batch_size = 32
    ff_dim = 5
    embed_dim = 5
    adam_init = 0.1
    # sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
    sample_size = 100
    num_survivors = 1000
    experiment_name = "dice4el"
    outcome_of_interest = None

    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    if CONFIG_IS_FRESH:
        reader = OutcomeTrafficShortReader(debug=True, mode=TaskModes.OUTCOME_EXTENSIVE_DEPRECATED).init_log(True).init_meta(True)
        reader.save(True)
    ds_name = OutcomeTrafficShortReader.__name__
    reader: OutcomeTrafficShortReader = OutcomeTrafficShortReader.load(PATH_READERS / ds_name)

    train_dataset = reader.get_dataset(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size)
    val_dataset = reader.get_dataset(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size)
    test_dataset = reader.get_test_dataset(ds_mode=DatasetModes.TEST, ft_mode=ft_mode)
    pname = ds_name.replace('Reader', 'Predictor')
    if CONFIG_IS_FRESH:
        predictor = PModel(name=pname, ff_dim=ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode)
        runner = PRunner(predictor, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)

    predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), compile=False)
    print("PREDICTOR")
    predictor.summary()

    vocab_len = reader.vocab_len
    max_len = reader.max_len
    default_mrate = MutationRate(*MUTATION_RATE_STAGE_3)
    feature_len = reader.feature_len  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.init_metrics(list(reader.feature_info.idx_discrete.values()), list(reader.feature_info.idx_continuous.values()))}
    # initiator = Initiator

    tr_cases, cf_cases, fa_cases = get_all_data(reader, ft_mode=ft_mode, fa_num=4)
    # fa_cases = get_d4el_data(reader)
    

    dist = DistributionConfig.registry(
        # tprobs=[MarkovChainProbability()],
        # eprobs=[
        #     EmissionProbIndependentFeatures(),
        # ],
    )[0]
    all_measure_configs = MeasureConfig.registry()
    data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, dist)

    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, all_measure_configs[0])

    # EVO GENERATOR

    all_evo_configs = create_combinations(0.1, default_mrate, evaluator)

    all_evo_configs = all_evo_configs[:2] if DEBUG_QUICK_MODE else all_evo_configs
    evo_wrappers = [
        build_evo_wrapper(
            ft_mode,
            top_k,
            sample_size,
            num_survivors,
            num_iterations,
            vocab_len,
            max_len,
            feature_len,
            predictor,
            evaluator,
            evo_config,
        ) for evo_config in all_evo_configs
    ] if not DEBUG_SKIP_EVO else []

    vae_wrapper = []

    casebased_wrappers = [build_cb_wrapper(
        ft_mode,
        top_k,
        sample_size,
        vocab_len,
        max_len,
        feature_len,
        tr_cases,
        predictor,
        evaluator,
    )] if not DEBUG_SKIP_CB else []

    randsample_wrapper = [build_rng_wrapper(
        ft_mode,
        top_k,
        sample_size,
        vocab_len,
        max_len,
        feature_len,
        predictor,
        evaluator,
    )] if not DEBUG_SKIP_RNG else []

    # for wrapper in all_wrappers:
    #     sorted_cases = wrapper.generate(fa_cases)

    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name

    experiment = ExperimentStatistics(idx2vocab=None)

    all_wrappers: List[GeneratorWrapper] = list(it.chain(*[vae_wrapper, casebased_wrappers, randsample_wrapper, evo_wrappers]))

    print(f"Computing {len(all_wrappers)} models")

    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name
    overall_folder_path = PATH_RESULTS / "bkp"
    if not overall_folder_path.exists():
        os.makedirs(overall_folder_path)

    # Parallel(backend='threading', n_jobs=4)(delayed(run_experiment)(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, err_log, exp_num, wrapper)
    #                                         for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)))
    all_results = []
    for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)):
        cf_results = wrapper.generate(fa_cases)
        for fa_id, (cf, fa) in enumerate(zip(cf_results, fa_cases)):
            fa_events, fa_features, fa_llh, _ = fa.all
            fa_events, fa_features, fa_llh = [np.repeat(el, len(cf), axis=0) for el in [fa_events, fa_features, fa_llh]]
            res = Cases(fa_events, fa_features, fa_llh)
            all_results.append({"name": wrapper.short_name, "cf": cf, "fa": res, "fa_id":fa_id})
            pickle.dump(all_results, io.open(PATH_RESULTS / f"results_traffic.pkl", "wb"))
            print("Got everything")


    print("TEST SIMPE STATS")
    print(experiment)
    print("")
    # experiment.data.to_csv(PATH_RESULTS / f"experiment_{experiment_name}_results.csv", index=False, line_terminator='\n')

    # overall_folder_path = PATH_RESULTS / "bkp"
    #
    # # Parallel(backend='threading', n_jobs=4)(delayed(run_experiment)(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, err_log, exp_num, wrapper)
    # #                                         for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)))

    # for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)):
    #     if not PATH_RESULTS.exists():
    #         os.makedirs(PATH_RESULTS/wrapper.name)
    # results = wrapper.generate(fa_cases)
    # events, features = results.cases

    # run_experiment(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, err_log, exp_num, wrapper)

    #
    print("TEST SIMPE STATS")
    # print(experiment)
    # print("")
    # experiment.data.to_csv(PATH_RESULTS / f"experiment_{experiment_name}_results.csv", index=False, line_terminator='\n')

    print("DONE")