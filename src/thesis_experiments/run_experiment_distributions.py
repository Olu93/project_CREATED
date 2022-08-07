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
import numpy as np
import time
from thesis_commons.config import DEBUG_USE_MOCK, READER
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import BayesianDistFeatures1, ChiSqEmissionProbFeatures, DataDistribution, DefaultEmissionProbFeatures, DistributionConfig, EmissionProbGroupedDistFeatures, EmissionProbIndependentFeatures, EmissionProbability, MarkovChainProbability
from thesis_commons.model_commons import GeneratorWrapper, TensorflowModelMixin
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_commons.representations import Cases, EvaluatedCases, MutationRate, Viabilities
from thesis_commons.statistics import ExperimentStatistics, StatCases, StatInstance, StatIteration, StatRow, StatRun
from thesis_experiments.commons import build_cb_wrapper, build_evo_wrapper, build_rng_wrapper, build_vae_wrapper, run_experiment
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies import evolutionary_operations
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader, OutcomeBPIC12Reader25
from thesis_readers.helper.helper import get_all_data, get_even_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from joblib import Parallel, delayed

DEBUG_QUICK_MODE = 0


def compute_stats(case_type: str, cases: Cases, data_distribution: DataDistribution, evaluator: ViabilityMeasure, fa_case: Cases):
    viabilities = evaluator.compute(fa_case, cases)
    t_probs, e_probs = data_distribution.compute_probability(cases)
    measure_types = viabilities.Measures
    prob: np.ndarray = None
    trnp: np.ndarray = None
    emip: np.ndarray = None
    case: Cases = None
    spars = viabilities.get(measure_types.SPARCITY)
    simls = viabilities.get(measure_types.SIMILARITY)
    dllhs = viabilities.get(measure_types.DATA_LLH)
    ollhs = viabilities.get(measure_types.OUTPUT_LLH)
    mllhs = viabilities.get(measure_types.MODEL_LLH)
    viabs = viabilities.get(measure_types.VIABILITY)
    probs = t_probs * e_probs
    aggr_probs, aggr_trnps, aggr_emips = probs.prod(-1, keepdims=True), t_probs.prod(-1, keepdims=True), e_probs.prod(-1, keepdims=True)
    true_outcomes = cases.outcomes * 1
    zipper = zip(cases, aggr_probs, aggr_trnps, aggr_emips, spars, simls, dllhs, ollhs, mllhs, viabs, true_outcomes)
    iteration = StatIteration()
    for case, prob, tprob, eprob, spar, siml, dllh, ollh, mllh, viab, true_o in tqdm(zipper, total=len(cases), desc=f"{case_type}"):
        row = StatRow()
        iteration = iteration.attach('case_origin', case_type)

        pred_o = (mllh > 0.5) * 1

        row = row.attach('prob', prob[0]).attach('tprob', tprob[0]).attach('eprob', eprob[0])
        row = row.attach('sparsity', spar[0]).attach('similarity', siml[0]).attach('feasibility', dllh[0]).attach('delta', ollh[0]).attach('viability', viab[0])
        row = row.attach('pred_score', mllh[0]).attach('true_outcome', true_o[0]).attach('pred_outcome', pred_o[0])
        row = row.attach('event', case.events[0])
        iteration = iteration.append(row)
        # instance.append(iteration).attach('dist_type', type(data_distribution))
    return iteration


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    num_iterations = 5 if DEBUG_QUICK_MODE else 50
    k_fa = 50
    experiment_name = "distributions"
    outcome_of_interest = None
    
    ds_name = READER
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    reader:AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
    predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()     
    
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    feature_len = reader.feature_len  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)

    tr_cases, cf_cases, _ = get_all_data(reader, ft_mode=ft_mode, fa_filter_lbl=outcome_of_interest, tr_num=1000, cf_num=1000)
    fa_cases = get_even_data(reader, ft_mode=ft_mode, fa_num=k_fa)


    all_measure_configs = MeasureConfig.registry()


    all_dist_configs = DistributionConfig.registry(
        tprobs=[MarkovChainProbability()],
        eprobs=[EmissionProbIndependentFeatures(), ChiSqEmissionProbFeatures(), DefaultEmissionProbFeatures(), EmissionProbGroupedDistFeatures()],
    )
    
    experiment = ExperimentStatistics()
    run = StatRun()
    experiment.append(run)
    all_distributions = [DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, config) for config in all_dist_configs]
    ddist: DataDistribution = None
    for ddist in tqdm(all_distributions, total=len(all_distributions), desc="DistType"):    
        evaluator = ViabilityMeasure(vocab_len, max_len, ddist, predictor, all_measure_configs[0])
        instance = StatInstance()
        instance.attach('ddist', ddist.get_config()).attach('evaluator', evaluator.get_config())
        true_cases = tr_cases
        sampled_cases = ddist.sample(len(tr_cases))
    
        for fa_case in tqdm(fa_cases, total=len(fa_cases), desc="Instance"):
            iteration1 = compute_stats("true_cases", true_cases, ddist, evaluator, fa_case)
            iteration2 = compute_stats("sampled_cases", sampled_cases, ddist, evaluator, fa_case)
            instance = instance.append(iteration1).append(iteration2)
        run.append(instance)

    results = experiment.data
    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name
    overall_folder_path = PATH_RESULTS
    if not overall_folder_path.exists():
        os.makedirs(overall_folder_path)
    results.to_csv(overall_folder_path / ('experiment_' + experiment_name + '_results.csv'), index=None)
    
    print("DONE")