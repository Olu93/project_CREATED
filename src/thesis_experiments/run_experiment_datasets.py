import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO, Tuple
import traceback
import itertools as it
import tensorflow as tf

from thesis_readers.readers.OutcomeReader import OutcomeReader

keras = tf.keras
from keras import models
from tqdm import tqdm
import time
from thesis_commons.config import DEBUG_USE_MOCK, MAX_ITER_STAGE_3, MUTATION_RATE_STAGE_3, READER
from thesis_commons.constants import (ALL_DATASETS, PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statistics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_experiments.commons import build_cb_wrapper, build_evo_wrapper, build_rng_wrapper, build_smpl_wrapper, build_vae_wrapper, build_vae_wrapper2, run_experiment
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader
from thesis_readers.helper.helper import get_all_data, get_even_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed

DEBUG_QUICK_MODE = 0
DEBUG_SKIP_VAE = 1
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 0
DEBUG_SKIP_RNG = 0
DEBUG_SKIP_SBG = 0
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True


# def create_combinations(erate: float, default_mrate: MutationRate, evaluator: ViabilityMeasure):
#     all_evo_configs = []
#     all_evo_configs.append(
#         evolutionary_operations.EvoConfigurator(
#             evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
#             evolutionary_operations.RouletteWheelSelector(),
#             evolutionary_operations.OnePointCrosser(),
#             evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
#             evolutionary_operations.FittestSurvivorRecombiner(),
#         ))
#     all_evo_configs.append(
#         evolutionary_operations.EvoConfigurator(
#             evolutionary_operations.CaseBasedInitiator().set_vault(evaluator.data_distribution),
#             evolutionary_operations.ElitismSelector(),
#             evolutionary_operations.TwoPointCrosser(),
#             evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
#             evolutionary_operations.BestBreedRecombiner(),
#         ))
#     all_evo_configs.append(
#         evolutionary_operations.EvoConfigurator(
#             evolutionary_operations.SamplingBasedInitiator().set_data_distribution(evaluator.data_distribution),
#             evolutionary_operations.ElitismSelector(),
#             evolutionary_operations.OnePointCrosser(),
#             evolutionary_operations.SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(default_mrate).set_edit_rate(None),
#             evolutionary_operations.RankedRecombiner(),
#         ))
#     return all_evo_configs
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
    k_fa = 3
    top_k = 10 if DEBUG_QUICK_MODE else 50
    # sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
    sample_size = 100
    num_survivors = 1000
    outcome_of_interest = 0
    default_mrate = MutationRate(*MUTATION_RATE_STAGE_3)
    measure_mask = MeasureMask(True, True, True, True)

    # ====================================================================================================================================================
    # ====================================================================================================================================================
    # ====================================================================================================================================================
    pairs: List[Tuple[OutcomeReader, TensorflowModelMixin, TensorflowModelMixin]] = []
    for ds_name in ALL_DATASETS:
        try:
            print("READER")
            reader: OutcomeReader = OutcomeReader.load(PATH_READERS / ds_name)
            custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
            custom_objects_generator = {
                obj.name: obj
                for obj in Generator.init_metrics(list(reader.feature_info.idx_discrete.values()), list(reader.feature_info.idx_continuous.values()))
            }
            print(f"Loaded {reader.name}")
            print("PREDICTOR")
            predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), custom_objects=custom_objects_predictor)
            print(f"Loaded {predictor.name}")
            predictor.summary()
            print("GENERATOR")
            generator: TensorflowModelMixin = None #models.load_model(PATH_MODELS_GENERATORS / ds_name.replace('Reader', 'Generator'), custom_objects=custom_objects_generator)
            # print(f"Loaded {generator.name}")
            # generator.summary()
            pairs.append((reader, predictor, generator))

        except Exception as e:
            print(f"Something went wrong loading {ds_name}: {e}")

    for reader, predictor, generator in pairs:
        experiment_name = "datasets"
        vocab_len = reader.vocab_len
        max_len = reader.max_len
        feature_len = reader.feature_len  # TODO: Change to function which takes features and extracts shape
        tr_cases, cf_cases, _ = get_all_data(reader, ft_mode=ft_mode)
        fa_cases = get_even_data(reader, ft_mode=ft_mode, fa_num=k_fa)
        all_measure_configs = MeasureConfig.registry()
        data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, DistributionConfig.registry()[0])

        evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, all_measure_configs[0])
        all_evo_configs = create_combinations(0.2, default_mrate, evaluator)

        # EVO GENERATOR
        run_meta = {
            "reader":reader.name,
            "max_len": reader.max_len,
            "virtual_max_len": reader.virual_max_len
        }

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

        sampling_wrapper = [build_smpl_wrapper(
            ft_mode,
            top_k,
            sample_size,
            vocab_len,
            max_len,
            feature_len,
            predictor,
            evaluator,
        )] if not DEBUG_SKIP_SBG else []


        experiment = ExperimentStatistics(idx2vocab=None)

        all_wrappers: List[GeneratorWrapper] = list(it.chain(*[evo_wrappers, vae_wrapper, casebased_wrappers, randsample_wrapper, sampling_wrapper]))

        print(f"Computing {len(all_wrappers)} models")

        PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name / reader.name
        overall_folder_path = PATH_RESULTS / "bkp"
        if not overall_folder_path.exists():
            os.makedirs(overall_folder_path)

        for exp_num, wrapper in tqdm(enumerate(all_wrappers), desc="Stats Run", total=len(all_wrappers), file=sys.stdout):
            run_experiment(experiment_name, measure_mask, fa_cases, experiment, overall_folder_path, None, exp_num, wrapper, run_meta=run_meta)

        
        print("\nEXPERIMENT 1 DONE\n")
        print(experiment)

        print("\nSAVE\n")
        experiment.data.to_csv(PATH_RESULTS / f"experiment_{experiment_name}_results.csv", index=False, line_terminator='\n')

        print("DONE")