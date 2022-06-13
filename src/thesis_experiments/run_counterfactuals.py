import os

import tensorflow as tf
from tqdm import tqdm

from thesis_commons.config import DEBUG_USE_MOCK, Reader
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_RESULTS_MODELS_OVERALL)
from thesis_commons.functions import get_all_data
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, MutationRate
from thesis_commons.statististics import ExperimentStatistics, ResultStatistics
from thesis_generators.generators.baseline_wrappers import (CaseBasedGeneratorWrapper, RandomGeneratorWrapper)
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
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureMask, ViabilityMeasure)

DEBUG_QUICK_MODE = 1
DEBUG_SKIP_VAE = 1
DEBUG_SKIP_EVO = 0
DEBUG_SKIP_CB = 1
DEBUG_SKIP_RNG = 1
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True


def generate_stats(stats: ResultStatistics, measure_mask, fa_cases, simple_vae_generator, simple_evo_generator, case_based_generator, rng_sample_generator):
    if simple_vae_generator is not None:
        stats = stats.update(model=simple_vae_generator, data=fa_cases, measure_mask=measure_mask)
    if simple_evo_generator is not None:
        stats = stats.update(model=simple_evo_generator, data=fa_cases, measure_mask=measure_mask)
    if case_based_generator is not None:
        stats = stats.update(model=case_based_generator, data=fa_cases, measure_mask=measure_mask)
    if rng_sample_generator is not None:
        stats = stats.update(model=rng_sample_generator, data=fa_cases, measure_mask=measure_mask)
    return stats


def build_vae_generator(top_k, sample_size, custom_objects_generator, predictor, evaluator):
    simple_vae_generator = None
    # VAE GENERATOR
    # TODO: Think of reversing cfs
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    vae_generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    vae_generator.summary()
    simple_vae_generator = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)
    return simple_vae_generator


def build_evo_generator(ft_mode, top_k, sample_size, mrate, vocab_len, max_len, feature_len, predictor, evaluator):
    evo_generator = SimpleEvolutionStrategy(max_iter=10 if DEBUG_QUICK_MODE else 100,
                                            evaluator=evaluator,
                                            ft_mode=ft_mode,
                                            vocab_len=vocab_len,
                                            max_len=max_len,
                                            feature_len=feature_len,
                                            mutation_rate=mrate,
                                            edit_rate=0.1)
    simple_evo_generator = SimpleEvoGeneratorWrapper(predictor=predictor, generator=evo_generator, evaluator=evaluator, top_k=top_k, sample_size=sample_size)

    return simple_evo_generator


if __name__ == "__main__":
    # combs = MeasureMask.get_combinations()
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    k_fa = 3
    top_k = 10 if not DEBUG_QUICK_MODE else 50
    sample_size = max(top_k, 1000) if not DEBUG_QUICK_MODE else max(top_k, 250)
    outcome_of_interest = 1
    reader: AbstractProcessLogReader = Reader.load()
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    default_mrate = MutationRate(0.01, 0.3, 0.3, 0.3)
    feature_len = reader.num_event_attributes  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.init_metrics()}

    tr_cases, cf_cases, fa_cases = get_all_data(reader, ft_mode=ft_mode, fa_num=k_fa, fa_filter_lbl=outcome_of_interest)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    evaluator = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor)

    # EVO GENERATOR

    simple_evo_generator = build_evo_generator(ft_mode, top_k, sample_size, default_mrate, vocab_len, max_len, feature_len, predictor, evaluator) if not DEBUG_SKIP_EVO else None
    simple_vae_generator = build_vae_generator(top_k, sample_size, custom_objects_generator, predictor, evaluator) if not DEBUG_SKIP_VAE else None

    cbg_generator = CaseBasedGeneratorModel(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, top_k=top_k,
                                                     sample_size=sample_size) if not DEBUG_SKIP_CB else None

    rng_generator = RandomGeneratorModel(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    rng_sample_generator = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, top_k=top_k,
                                                  sample_size=sample_size) if not DEBUG_SKIP_RNG else None

    if not DEBUG_SKIP_SIMPLE_EXPERIMENT:
        stats = ResultStatistics(reader.idx2vocab)

        stats = generate_stats(stats, measure_mask, fa_cases, simple_vae_generator, simple_evo_generator, case_based_generator, rng_sample_generator)

        print("TEST SIMPE STATS")
        print(stats)
        print("")
        print(stats.data.iloc[:, :-2])
        print("")
        stats.data.to_csv(PATH_RESULTS_MODELS_OVERALL / "cf_generation_results.csv", index=False, line_terminator='\n')

    if not DEBUG_SKIP_MASKED_EXPERIMENT:
        print("RUN ALL MASK CONFIGS")
        all_stats = ExperimentStatistics()
        mask_combs = MeasureMask.get_combinations()
        pbar = tqdm(enumerate(mask_combs), total=len(mask_combs))
        for idx, mask_comb in pbar:
            tmp_mask: MeasureMask = mask_comb
            pbar.set_description(f"MASK_CONFIG {list(tmp_mask.to_num())}", refresh=True)
            tmp_stats = generate_stats(mask_comb, fa_cases, simple_evo_generator, case_based_generator, rng_sample_generator, simple_vae_generator)
            all_stats.update(idx, tmp_stats)

        print("EXPERIMENTAL RESULTS")
        print(all_stats._data)
        all_stats._data.to_csv(PATH_RESULTS_MODELS_OVERALL / "cf_generation_results_experiment.csv", index=False, line_terminator='\n')

        print("DONE")