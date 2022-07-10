import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO
import traceback
import itertools as it
import tensorflow as tf
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models
from tqdm import tqdm
import time
from thesis_commons.config import DEBUG_QUICK_EVO_MODE, DEBUG_USE_MOCK
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statististics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_generators.generators.baseline_wrappers import (CaseBasedGeneratorWrapper, RandomGeneratorWrapper)
from thesis_generators.generators.evo_wrappers import EvoGeneratorWrapper
from thesis_generators.generators.vae_wrappers import SimpleVAEGeneratorWrapper
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.baselines.baseline_search import CaseBasedGenerator, RandomGenerator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_generators.models.evolutionary_strategies.evolutionary_strategy import EvolutionaryStrategy
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader
from thesis_readers.helper.helper import get_all_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)

def build_vae_wrapper(top_k, sample_size, custom_objects_generator, predictor, evaluator):
    simple_vae_wrapper = None
    # VAE GENERATOR
    # TODO: Think of reversing cfs
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    vae_generator: TensorflowModelMixin = models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    vae_generator.summary()
    simple_vae_wrapper = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)
    return simple_vae_wrapper


def build_evo_wrapper(ft_mode, top_k, sample_size, survival_thresh, max_iter, vocab_len, max_len, feature_len, predictor: TensorflowModelMixin, evaluator: ViabilityMeasure,
                      evo_config: evolutionary_operations.EvoConfigurator):

    evo_strategy = EvolutionaryStrategy(
        max_iter=max_iter,
        evaluator=evaluator,
        operators=evo_config,
        ft_mode=ft_mode,
        vocab_len=vocab_len,
        max_len=max_len,
        survival_thresh=survival_thresh,
        feature_len=feature_len,
    )
    evo_wrapper = EvoGeneratorWrapper(predictor=predictor, generator=evo_strategy, evaluator=evaluator, top_k=top_k, sample_size=sample_size)
    return evo_wrapper


def build_cb_wrapper(ft_mode, top_k, sample_size, vocab_len, max_len, feature_len, tr_cases, predictor, evaluator):
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    casebased_wrapper = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)

    return casebased_wrapper


def build_rng_wrapper(ft_mode, top_k, sample_size, vocab_len, max_len, feature_len, predictor, evaluator):
    rng_generator = RandomGenerator(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    randsample_wrapper = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)

    return randsample_wrapper


def save_specific_model_results(experiment_name: str, wrapper: GeneratorWrapper, extra_name=""):
    specific_folder_path = PATH_RESULTS_MODELS_SPECIFIC / experiment_name / wrapper.name
    if not specific_folder_path.exists():
        os.makedirs(specific_folder_path)
    specific_file_name = wrapper.short_name + ("_" + extra_name if extra_name else "")
    save_path = wrapper.save_statistics(specific_folder_path, specific_file_name)
    save_report = f"{'SUCCESS' if save_path else 'FAIL'}: Statistics of {specific_file_name} in {save_path}\n"
    print(save_report)


def save_bkp_model_results(experiment: ExperimentStatistics, overall_folder_path: pathlib.Path, exp_num):
    experiment.data.to_csv(overall_folder_path / f"backup_{exp_num}.csv", index=False, line_terminator='\n')


def attach_results_to_stats(measure_mask: MeasureMask, experiment: ExperimentStatistics, wrapper: GeneratorWrapper, runs: StatRun, instances: StatInstance, results, config):
    start_time = time.time()
    for result_instance in results:
        instances = instances.append(StatCases().attach(result_instance))

    duration = time.time() - start_time
    duration_time = datetime.timedelta(seconds=duration)
    runs = runs.append(instances).attach("mask", measure_mask.to_binstr())
    runs = runs.attach('duration', str(duration_time)).attach('duration_sec', duration)
    runs = runs.attach('short_name', wrapper.short_name).attach('full_name', wrapper.full_name)
    runs = runs.attach(None, config)
    experiment.append(runs)
    sys.stdout.flush()


def run_experiment(experiment_name: str, measure_mask: MeasureMask, fa_cases: Cases, experiment: ExperimentStatistics, overall_folder_path: pathlib.Path, err_log: TextIO, exp_num,
                   wrapper, extra_name=""):
    try:
        runs = StatRun()
        instances = StatInstance()
        wrapper: GeneratorWrapper = wrapper.set_measure_mask(measure_mask)
        results = wrapper.generate(fa_cases)
        config = wrapper.get_config()
        attach_results_to_stats(measure_mask, experiment, wrapper, runs, instances, results, config)
        save_bkp_model_results(experiment, overall_folder_path, exp_num)
        save_specific_model_results(experiment_name, wrapper, extra_name)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        fname = exc_tb.tb_frame.f_code.co_filename
        err = f"\nRUN FAILED! - {e} - {wrapper.full_name}\n\nDETAILS:\nType:{exc_type} File:{fname} Line:{exc_tb.tb_lineno}"
        print(err + "\n" + f"{traceback.format_exc()}")
        err_log.write(err + "\n")
