import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO
import traceback
import itertools as it
import tensorflow as tf

keras = tf.keras
from keras import models
from tqdm import tqdm
import time
from thesis_commons.random import random
from thesis_commons.config import DEBUG_USE_MOCK, MAX_ITER_STAGE_SPECIAL, MUTATION_RATE_STAGE_1, POPULATION_SIZE, READER, SAMPLE_SIZE
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statistics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_experiments.commons import build_cb_wrapper, build_evo_wrapper, build_rng_wrapper, build_vae_wrapper, run_experiment
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import *
from thesis_readers.helper.helper import get_all_data, get_even_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed

DEBUG_QUICK_MODE = 0
DEBUG_SKIP_VAE = 1
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 1
DEBUG_SKIP_RNG = 1
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True


def create_combinations(erate: float, mrate: MutationRate, evaluator: ViabilityMeasure):
    initiators = [
        # evolutionary_operations.FactualInitiator(),
        evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
        # evolutionary_operations.SamplingBasedInitiator().set_data_distribution(evaluator.measures.dllh.data_distribution),
    ]
    selectors = [
        # evolutionary_operations.RouletteWheelSelector(),
        evolutionary_operations.ElitismSelector(),
        # evolutionary_operations.TournamentSelector(),
    ]
    crossers = [
        evolutionary_operations.OnePointCrosser(),
    ]
    mutators = [evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(mrate).set_edit_rate(erate)]
    recombiners = [
        evolutionary_operations.FittestSurvivorRecombiner(),
        evolutionary_operations.BestBreedRecombiner(),
    ]
    combos = it.product(initiators, selectors, crossers, mutators, recombiners)
    return combos


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    all_iterations = 5 if DEBUG_QUICK_MODE else MAX_ITER_STAGE_SPECIAL
    # all_iterations = [5] if DEBUG_QUICK_MODE else [2, 5]
    top_k = 10 if DEBUG_QUICK_MODE else 50
    k_fa = 2

    # sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
    sample_size = SAMPLE_SIZE
    num_survivors = POPULATION_SIZE
    experiment_name = "evolutionary_iterations_low"
    outcome_of_interest = None

    ds_name = READER
    reader: AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
    predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), compile=False)
    print("PREDICTOR")
    predictor.summary()

    vocab_len = reader.vocab_len
    max_len = reader.max_len
    default_mrate = MutationRate(*MUTATION_RATE_STAGE_1)
    feature_len = reader.feature_len  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)
    # initiator = Initiator

    tr_cases, cf_cases, _ = get_all_data(reader, ft_mode=ft_mode)
    fa_cases = get_even_data(reader, ft_mode=ft_mode, fa_num=k_fa)

    all_measure_configs = MeasureConfig.registry()
    data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, DistributionConfig.registry()[0])

    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, all_measure_configs[0])

    # EVO GENERATOR

    # all_mutation_rates = [0.1]

    # combos = create_combinations(None, default_mrate, evaluator)

    all_evo_configs = []
    all_evo_configs.append(
        evolutionary_operations.EvoConfigurator(
            evolutionary_operations.RandomInitiator(),
            evolutionary_operations.RouletteWheelSelector(),
            evolutionary_operations.TwoPointCrosser(),
            evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
            evolutionary_operations.RankedRecombiner(),
        ))

    all_evo_configs.append(
        evolutionary_operations.EvoConfigurator(
            evolutionary_operations.RandomInitiator(),
            evolutionary_operations.TournamentSelector(),
            evolutionary_operations.TwoPointCrosser(),
            evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
            evolutionary_operations.RankedRecombiner(),
        ))

    evo_wrappers = [
        build_evo_wrapper(
            ft_mode,
            top_k,
            sample_size,
            num_survivors,
            all_iterations,
            vocab_len,
            max_len,
            feature_len,
            predictor,
            evaluator,
            evo_config,
        ) for evo_config in all_evo_configs
    ] if not DEBUG_SKIP_EVO else []

    vae_wrapper = []
    casebased_wrappers = []
    randsample_wrapper = []

    experiment = ExperimentStatistics(idx2vocab=None)

    all_wrappers: List[GeneratorWrapper] = list(it.chain(*[vae_wrapper, casebased_wrappers, randsample_wrapper, evo_wrappers]))

    print(f"Computing {len(all_wrappers)} models")

    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name
    overall_folder_path = PATH_RESULTS / "bkp"
    if not overall_folder_path.exists():
        os.makedirs(overall_folder_path)
    
    # Parallel(backend='threading', n_jobs=4)(delayed(run_experiment)(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, err_log, exp_num, wrapper)
    #                                         for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)))

    for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)):
        run_experiment(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, None, exp_num, wrapper)

    
    print("TEST SIMPE STATS")
    print(experiment)
    print("")
    experiment.data.to_csv(PATH_RESULTS / f"experiment_{experiment_name}_results.csv", index=False, line_terminator='\n')

    print("DONE")