import os

import numpy as np
import tensorflow as tf

keras = tf.keras
from keras import backend as K, losses, metrics, utils, layers, optimizers, models
from thesis_commons.config import DEBUG_USE_MOCK
from thesis_commons.constants import (PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, PATH_READERS)
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_readers import Reader
from thesis_readers.helper.helper import get_all_data, get_even_data
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import FeatureModes, TaskModes
from thesis_commons.representations import Cases
from thesis_generators.generators.baseline_wrappers import \
    CaseBasedGeneratorWrapper
from thesis_generators.models.baselines.baseline_search import CaseBasedGenerator
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask, ViabilityMeasure)
from thesis_viability.dice4el_measure.measures import Dice4ELCategoryChangeMeasure, Dice4ELDiversityMeasure, Dice4ELPlausibilityMeasure, Dice4ELProximityMeasure, Dice4ELSparcityMeasure

DEBUG = True

# TODO: Make viability measure train data be a Cases object
if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 50
    ft_mode = FeatureModes.FULL
    epochs = 50
    topk = 5
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}

    # generative_reader = GenerativeDataset(reader)
    ds_name = "OutcomeBPIC12Reader25"
    reader: AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
    predictor: TensorflowModelMixin = models.load_model(PATH_MODELS_PREDICTORS / ds_name.replace('Reader', 'Predictor'), compile=False)
    print("PREDICTOR")
    predictor.summary()

    vocab_len = reader.vocab_len
    max_len = reader.max_len
    feature_len = reader.feature_len  # TODO: Change to function which takes features and extracts shape
    measure_mask = MeasureMask(True, True, True, True)

    tr_cases, cf_cases, _ = get_all_data(reader, ft_mode=ft_mode)
    fa_cases = get_even_data(reader, ft_mode=ft_mode, fa_num=2)

    measure = MeasureConfig.registry(
        sparsity=[Dice4ELSparcityMeasure()],
        similarity=[Dice4ELProximityMeasure()],
        dllh = [Dice4ELPlausibilityMeasure()],
        ollh = [Dice4ELCategoryChangeMeasure()],
    )
    # measure = MeasureConfig.registry( )[0]
    dist = DistributionConfig.registry()[0]
    data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, dist)

    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, measure[0])
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)

    cf_cases, _ = cbg_generator.predict(cf_cases)
    # all_measure_configs = MeasureConfig.registry()
    # data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader.feature_info, DistributionConfig.registry()[0])
    print("Test SIMPLE")
    for measure_cnf in measure:
        viability = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, measure_cnf)

        viability_values = viability(fa_cases, cf_cases)
        print("VIABILITIES")
        print(viability_values)

    print("Test END-TO-END")
    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, measure[0])
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    print(case_based_generator.generate(fa_cases))

    print("Test MODULARITY")
    evaluator = ViabilityMeasure(vocab_len, max_len, data_distribution, predictor, measure[0])
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor,
                                                     generator=cbg_generator,
                                                     evaluator=evaluator,
                                                     topk=topk,
                                                     measure_mask=MeasureMask(False, True, False, True),
                                                     sample_size=max(topk, 1000))
    print(case_based_generator.generate(fa_cases))
    print("DONE")