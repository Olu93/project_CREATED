import os

import numpy as np
import tensorflow as tf

from thesis_commons.config import DEBUG_USE_MOCK, Reader
from thesis_commons.constants import (PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS)
from thesis_readers.helper.helper import get_all_data
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import FeatureModes, TaskModes
from thesis_commons.representations import Cases
from thesis_generators.generators.baseline_wrappers import \
    CaseBasedGeneratorWrapper
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGenerator
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as Generator
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.viability.viability_function import (MeasureConfig, MeasureMask,
                                                           ViabilityMeasure)

DEBUG = True

# TODO: Make viability measure train data be a Cases object
if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader: AbstractProcessLogReader = Reader.load()
    epochs = 50
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    ft_mode = FeatureModes.FULL
    epochs = 50
    topk = 5
    feature_len = reader.num_event_attributes
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.init_metrics()}

    # generative_reader = GenerativeDataset(reader)
    tr_cases, cf_cases, fa_cases = get_all_data(reader, ft_mode=ft_mode, fa_num=5, fa_filter_lbl=1)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    generator: TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    generator.summary()

    cf_events, cf_features = generator.predict([np.repeat(fa_cases.events, len(cf_cases), axis=0), np.repeat(fa_cases.features, len(cf_cases), axis=0)])
    all_measure_configs = MeasureConfig.registry()
    
    for measure_cnf in all_measure_configs:
        viability = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor, measure_cnf)

        viability_values = viability(fa_cases, Cases(cf_events.astype(float), cf_features))
        print("VIABILITIES")
        print(viability_values)

    print("Test END-TO-END")
    evaluator = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor, all_measure_configs[0])
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    print(case_based_generator.generate(fa_cases))

    print("Test MODULAR VIABILITY")
    evaluator = ViabilityMeasure(vocab_len, max_len, tr_cases, predictor, all_measure_configs[0])
    cbg_generator = CaseBasedGenerator(tr_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor,
                                                     generator=cbg_generator,
                                                     evaluator=evaluator,
                                                     topk=topk,
                                                     measure_mask=MeasureMask(False, True, False, True),
                                                     sample_size=max(topk, 1000))
    print(case_based_generator.generate(fa_cases))
    print("DONE")