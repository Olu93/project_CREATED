import os

import tensorflow as tf
from tqdm import tqdm

from thesis_commons.config import DEBUG_USE_MOCK
from thesis_commons.constants import (PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS,
                                      PATH_RESULTS_COUNTERFACTUALS)
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases
from thesis_commons.statististics import ExperimentStatistics, ResultStatistics
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
from thesis_viability.viability.viability_function import (MeasureMask,
                                                           ViabilityMeasure)

DEBUG_SKIP_VAE = False
DEBUG_SKIP_SIMPLE_EXPERIMENT = False
DEBUG_SKIP_MASKED_EXPERIMENT = True

if DEBUG_USE_MOCK:
    from thesis_readers import OutcomeMockReader as Reader
else:
    from thesis_readers import OutcomeBPIC12Reader as Reader


def generate_stats(measure_mask, fa_cases, simple_vae_generator, simple_evo_generator, case_based_generator, rng_sample_generator):
    stats = ResultStatistics()
    if simple_vae_generator is not None:
        stats = stats.update(model=simple_vae_generator, data=fa_cases, measure_mask=measure_mask)
    if simple_evo_generator is not None:
        stats = stats.update(model=simple_evo_generator, data=fa_cases, measure_mask=measure_mask)
    if case_based_generator is not None:
        stats = stats.update(model=case_based_generator, data=fa_cases, measure_mask=measure_mask)
    if rng_sample_generator is not None:
        stats = stats.update(model=rng_sample_generator, data=fa_cases, measure_mask=measure_mask)
    return stats


def build_vae_generator(topk, custom_objects_generator, predictor, evaluator):
    simple_vae_generator = None
    if not DEBUG_SKIP_VAE:
        # VAE GENERATOR
        # TODO: Think of reversing cfs
        all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
        vae_generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
        print("GENERATOR")
        vae_generator.summary()
        simple_vae_generator = SimpleVAEGeneratorWrapper(predictor=predictor, generator=vae_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 100))
    return simple_vae_generator


if __name__ == "__main__":
    # combs = MeasureMask.get_combinations()
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    epochs = 50
    k_fa = 3
    topk = 10
    outcome_of_interest = 1
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    feature_len = reader.current_feature_len  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}

    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), cf_labels = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)

    fa_events, fa_features, fa_labels = fa_events[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_features[fa_labels[:, 0] == outcome_of_interest][:k_fa], fa_labels[
        fa_labels[:, 0] == outcome_of_interest][:k_fa]
    fa_cases = Cases(fa_events, fa_features, fa_labels)
    assert len(fa_cases) > 0, "Abort random selection failed"
    tr_cases = Cases(cf_events, cf_features, cf_labels)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    evaluator = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor)

    # EVO GENERATOR
    evo_generator = SimpleEvolutionStrategy(max_iter=100, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    cbg_generator = CaseBasedGeneratorModel(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    rng_generator = RandomGeneratorModel(evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)

    simple_vae_generator = build_vae_generator(topk, custom_objects_generator, predictor, evaluator)
    simple_evo_generator = SimpleEvoGeneratorWrapper(predictor=predictor, generator=evo_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    rng_sample_generator = RandomGeneratorWrapper(predictor=predictor, generator=rng_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))

    if not DEBUG_SKIP_SIMPLE_EXPERIMENT:

        stats = generate_stats(measure_mask, fa_cases, simple_vae_generator, simple_evo_generator, case_based_generator, rng_sample_generator)

        print("TEST SIMPE STATS")
        print(stats)
        print("")
        print(stats.data)
        print("")
        stats.data.to_csv(PATH_RESULTS_COUNTERFACTUALS / "cf_generation_results.csv")


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
        all_stats._data.to_csv(PATH_RESULTS_COUNTERFACTUALS / "cf_generation_results_experiment.csv")

        print("DONE")