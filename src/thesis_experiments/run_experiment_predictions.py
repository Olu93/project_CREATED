import datetime
import io
import os
import pathlib
import sys
from typing import List, TextIO, Tuple
import traceback
import itertools as it
import tensorflow as tf

keras = tf.keras
from keras import models
from tqdm import tqdm
import numpy as np
import time
from thesis_commons.config import DEBUG_USE_MOCK, READER
from thesis_commons.constants import (ALL_DATASETS, PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS, PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC)
from thesis_commons.distributions import ChiSqEmissionProbFeatures, DataDistribution, DefaultEmissionProbFeatures, DistributionConfig, EmissionProbIndependentFeatures, EmissionProbability, MarkovChainProbability
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
from thesis_readers.helper.helper import get_all_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from sklearn import metrics

DEBUG_QUICK_MODE = 0


def compute_stats(data_subset:str, cases: Cases, ds: AbstractProcessLogReader, predictor: TensorflowModelMixin):
    predictor_type = predictor.name
    y_trues = cases.outcomes
    y_preds = (predictor.predict(cases.cases) > 0.5)
    case: Cases = None
    iteration = StatIteration()
    accr = metrics.accuracy_score(y_trues, y_preds)
    bacc = metrics.balanced_accuracy_score(y_trues, y_preds)
    prec = metrics.precision_score(y_trues, y_preds)
    recl = metrics.recall_score(y_trues, y_preds)
    f1sc = metrics.f1_score(y_trues, y_preds)
    iteration.attach('subset', data_subset)
    iteration = iteration.attach('length', len(cases))
    iteration = iteration.attach('accuracy', accr).attach('balanced_accuracy', bacc)
    iteration = iteration.attach('precision', prec).attach('recall', recl).attach('f1', f1sc)
    zipper = zip(cases, y_trues, y_preds)
    for case, y_true, y_pred in tqdm(zipper, total=len(cases), desc=f"{predictor_type}"):
        row = StatRow()
        row = row.attach('event', case.events[0])
        row = row.attach('true_outcome', y_true[0])
        row = row.attach('pred_outcome', y_pred[0])
        iteration = iteration.append(row)
        # instance.append(iteration).attach('dist_type', type(data_distribution))
    return iteration


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL
    num_iterations = 5 if DEBUG_QUICK_MODE else 50
    k_fa = 2 if DEBUG_QUICK_MODE else 30
    experiment_name = "predictions"
    outcome_of_interest = None
    reader: AbstractProcessLogReader = Reader.load(PATH_READERS / READER)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    all_measure_configs = MeasureConfig.registry()

    pairs: List[Tuple[AbstractProcessLogReader, TensorflowModelMixin]] = []
    for ds_name in ALL_DATASETS: 
        try:
            print("READER")
            reader:AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
            print(f"Loaded {reader.name}")
            print("PREDICTOR")
            predictor:TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), custom_objects=custom_objects_predictor)
            print(f"Loaded {predictor.name}")
            predictor.summary()
            pairs.append((reader,predictor))
        except Exception as e:
            print(f"Something went wrong loading {ds_name}: {e}")
            

    experiment = ExperimentStatistics()
    run = StatRun()
    experiment.append(run)
    ds: AbstractProcessLogReader = None
    for ds, predictor in tqdm(pairs, total=len(pairs), desc="Dataset"):
        instance = StatInstance()
        ds_stats = ds.get_data_statistics()
        del ds_stats['starting_column_stats'] 
        instance.attach('predictor', predictor.name).attach('dataset', ds_stats)
        tr_cases, cf_cases, fa_cases = get_all_data(ds, ft_mode=ft_mode, fa_num=k_fa, fa_filter_lbl=outcome_of_interest)

        iteration1 = compute_stats("training", tr_cases, ds, predictor)
        iteration2 = compute_stats("validation", cf_cases, ds, predictor)
        iteration3 = compute_stats("test", fa_cases, ds, predictor)
        instance = instance.append(iteration1).append(iteration2).append(iteration3)
        run.append(instance)

    results = experiment.data
    PATH_RESULTS = PATH_RESULTS_MODELS_OVERALL / experiment_name
    overall_folder_path = PATH_RESULTS
    if not overall_folder_path.exists():
        os.makedirs(overall_folder_path)
    results.to_csv(overall_folder_path / ('experiment_' + experiment_name + '_results.csv'), index=None)
    err_log = io.open(f'error_{experiment_name}.log', 'w')
    print("DONE")