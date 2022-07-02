import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO
import traceback
import itertools as it
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm
import time
from thesis_commons.random import random
from thesis_commons.config import DEBUG_USE_MOCK
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statististics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_experiments.commons import build_cb_wrapper, build_evo_wrapper, build_rng_wrapper, build_vae_wrapper, run_experiment
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader
from thesis_readers.helper.helper import get_all_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed

DEBUG_QUICK_MODE = 1
DEBUG_SKIP_VAE = 1
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 1
DEBUG_SKIP_RNG = 1
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True

if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    max_iter = 5 if DEBUG_QUICK_MODE else 30
    k_fa = 1
    top_k = 10 if DEBUG_QUICK_MODE else 50
    # sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
    sample_size = 100 if DEBUG_QUICK_MODE else 1000
    experiment_name = "evolutionary_params"
    outcome_of_interest = None
    reader: AbstractProcessLogReader = Reader.load()
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    default_mrate = MutationRate(0.01, 0.3, 0.3, 0.3)
    feature_len = reader.num_event_attributes  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.init_metrics()}
    # initiator = Initiator

    tr_cases, cf_cases, fa_cases = get_all_data(reader, ft_mode=ft_mode, fa_num=k_fa, fa_filter_lbl=outcome_of_interest)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    all_measure_configs = MeasureConfig.registry()
    data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader._idx_distribution, DistributionConfig.registry()[0])

    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, all_measure_configs[0])

    # EVO GENERATOR

    all_mutation_rates = []
    for i in range(15):
        remainder = 1
        curr_list = []
        for j in range(4):
            val = random.uniform(0, remainder)
            curr_list.append(val)
            remainder -= val
        curr_list.append(remainder)
        random.shuffle(curr_list)
        all_mutation_rates.append(MutationRate(*curr_list))

    all_edit_rates = [0.1, 0.5, 0.9]

    initiators = [
        evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
    ]
    selectors = [
        evolutionary_operations.RouletteWheelSelector(),
    ]
    crossers = [
        evolutionary_operations.OnePointCrosser(),
    ]
    mutators = [
        evolutionary_operations.DataDistributionMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(mrate).set_edit_rate(erate)
        for mrate in all_mutation_rates for erate in all_edit_rates
    ]
    recombiners = [
        evolutionary_operations.FittestIndividualRecombiner(),
    ]

    combos = it.product(initiators, selectors, crossers, mutators, recombiners)
    all_evo_configs = [evolutionary_operations.EvoConfigurator(*cnf) for cnf in combos]

    all_evo_configs = all_evo_configs[:2] if DEBUG_QUICK_MODE else all_evo_configs


    evo_wrappers = [
        build_evo_wrapper(
            ft_mode,
            top_k,
            sample_size,
            int(sample_size * 0.5),
            max_iter,
            vocab_len,
            max_len,
            feature_len,
            predictor,
            evaluator,
            evo_config,
        ).set_extra_name(edit_rate=evo_config.mutator.edit_rate, mutation_rate="-".join([f"{v:.3f}" for v in evo_config.mutator.mutation_rate.to_dict().values()])) for evo_config in all_evo_configs 
    ] if not DEBUG_SKIP_EVO else []

    vae_wrapper =  []
    casebased_wrappers = []
    randsample_wrapper =  []

    experiment = ExperimentStatistics(idx2vocab=None)

    all_wrappers: List[GeneratorWrapper] = list(it.chain(*[vae_wrapper, casebased_wrappers, randsample_wrapper, evo_wrappers]))

    print(f"Computing {len(all_wrappers)} models")

    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name
    overall_folder_path = PATH_RESULTS / "bkp"
    if not overall_folder_path.exists():
        os.makedirs(overall_folder_path)
    err_log = io.open(f'error_{experiment_name}.log', 'w')

    for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers)):
        run_experiment(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, err_log, exp_num, wrapper)

    err_log.close()
    print("TEST SIMPE STATS")
    print(experiment)
    print("")
    experiment.data.to_csv(PATH_RESULTS / f"experiment_{experiment_name}_results.csv", index=False, line_terminator='\n')

    print("DONE")