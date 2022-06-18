import datetime
import io
import os
import sys
from typing import List

import tensorflow as tf
from tqdm import tqdm
import time
from thesis_commons.config import DEBUG_USE_MOCK, Reader
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_RESULTS_MODELS_OVERALL)
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statististics import ExperimentStatistics, StatCases, StatInstance, StatRun
from thesis_generators.generators.baseline_wrappers import (CaseBasedGeneratorWrapper, RandomGeneratorWrapper)
from thesis_generators.generators.evo_wrappers import EvoGeneratorWrapper
from thesis_generators.generators.vae_wrappers import SimpleVAEGeneratorWrapper
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGenerator
from thesis_generators.models.baselines.random_search import \
    RandomGenerator
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers.helper.helper import get_all_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)

DEBUG_QUICK_MODE = 1
DEBUG_SKIP_VAE = 0
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 1
DEBUG_SKIP_RNG = 0
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True




def build_vae_wrapper(top_k, sample_size, custom_objects_generator, predictor, evaluator):
    simple_vae_wrapper = None
    # VAE GENERATOR
    # TODO: Think of reversing cfs
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    vae_generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    vae_generator.summary()
    simple_vae_wrapper = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)
    return simple_vae_wrapper


def build_evo_wrapper(ft_mode, top_k, sample_size, mrate, vocab_len, max_len, feature_len, predictor: TensorflowModelMixin, evaluator: ViabilityMeasure,
                      evo_config: evolutionary_operations.EvoConfig):

    evo_strategy = EvolutionaryStrategy(
        max_iter=2 if DEBUG_QUICK_MODE else 100,
        evaluator=evaluator,
        operators=evo_config,
        ft_mode=ft_mode,
        vocab_len=vocab_len,
        max_len=max_len,
        feature_len=feature_len,
    )
    evo_wrapper = EvoGeneratorWrapper(predictor=predictor, generator=evo_strategy, evaluator=evaluator, top_k=top_k, sample_size=sample_size)
    return evo_wrapper


if __name__ == "__main__":
    # combs = MeasureMask.get_combinations()
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    k_fa = 3
    top_k = 10 if DEBUG_QUICK_MODE else 50
    sample_size = max(top_k, 100) if DEBUG_QUICK_MODE else max(top_k, 1000)
    outcome_of_interest = 1
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
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    all_measure_configs = MeasureConfig.registry()
    evaluator = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor, all_measure_configs[0])

    # EVO GENERATOR

    evo_configs = evolutionary_operations.EvoConfigurator.combinations(evaluator=evaluator, mutation_rate=default_mrate)
    evo_configs = evo_configs[:2] if DEBUG_QUICK_MODE else evo_configs
    evo_wrappers = [
        build_evo_wrapper(
            ft_mode,
            top_k,
            sample_size,
            default_mrate,
            vocab_len,
            max_len,
            feature_len,
            predictor,
            evaluator,
            evo_config,
        ) for evo_config in evo_configs
    ] if not DEBUG_SKIP_EVO else None

    vae_wrapper = build_vae_wrapper(top_k, sample_size, custom_objects_generator, predictor, evaluator) if not DEBUG_SKIP_VAE else None

    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    casebased_wrapper = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, top_k=top_k,
                                                  sample_size=sample_size) if not DEBUG_SKIP_CB else None

    rng_generator = RandomGenerator(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    randsample_wrapper = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, top_k=top_k,
                                                sample_size=sample_size) if not DEBUG_SKIP_RNG else None

    if not DEBUG_SKIP_SIMPLE_EXPERIMENT:
        experiment = ExperimentStatistics(idx2vocab=None)

        wrappers: List[GeneratorWrapper] = [vae_wrapper, casebased_wrapper, randsample_wrapper] + evo_wrappers
        all_wrappers = [wrapper for wrapper in wrappers if wrapper is not None]
        print(f"Computing {len(all_wrappers)} models")
        err_log = io.open('error.log', 'w')
        for wrapper in tqdm(all_wrappers, desc="Stats Run", total=len(all_wrappers)):
            try:
                start_time = time.time()
                runs = StatRun()
                instances = StatInstance()
                wrapper: GeneratorWrapper = wrapper.set_measure_mask(measure_mask)
                results = wrapper.generate(fa_cases)
                config = wrapper.get_config()
                for result_instance in results:
                    instances = instances.append(StatCases().attach(result_instance))

                duration = time.time() - start_time
                duration_time = datetime.timedelta(seconds=duration)
                runs = runs.attach("wrapper", wrapper.name).append(instances).attach("mask", measure_mask.to_binstr())
                runs = runs.attach('duration', str(duration_time)).attach('duration_sec', duration)
                runs = runs.attach(None, config)
                experiment.append(runs)
                sys.stdout.flush()
            except Exception as e:
                err = f"RUN FAILED! - {e} - {wrapper.config_name}"
                print(err)
                err_log.write(err + "\n")
        err_log.close()
        print("TEST SIMPE STATS")
        print(experiment)
        print("")
        experiment.data.to_csv(PATH_RESULTS_MODELS_OVERALL / "cf_generation_results.csv", index=False, line_terminator='\n')

    print("DONE")