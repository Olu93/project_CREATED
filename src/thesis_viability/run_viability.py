import os

import numpy as np
import pandas as pd
import tensorflow as tf

import thesis_commons.metric as metric
from thesis_commons.constants import (PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS)
from thesis_commons.functions import reverse_sequence_2, stack_data
from thesis_commons.libcuts import K, layers, losses
from thesis_commons.model_commons import TensorflowModelMixin
from thesis_commons.modes import (DatasetModes, FeatureModes, GeneratorModes,
                                  TaskModes)
from thesis_commons.representations import Cases
from thesis_generators.generators.baseline_wrappers import CaseBasedGeneratorWrapper
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_generators.models.baselines.casebased_heuristic import CaseBasedGeneratorModel
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as Generator
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
from thesis_viability.helper.base_distances import odds_ratio as dist
from thesis_viability.outcomellh.outcomllh_measure import \
    SummarizedNextActivityImprovementMeasureOdds as ImprovementMeasure
from thesis_viability.similarity.similarity_measure import SimilarityMeasure
from thesis_viability.sparcity.sparcity_measure import SparcityMeasure
from thesis_viability.viability.viability_function import MeasureMask, ViabilityMeasure

DEBUG = True

DEBUG_USE_MOCK = True
if DEBUG_USE_MOCK:
    from thesis_readers import OutcomeMockReader as Reader
else:
    from thesis_readers import OutcomeBPIC12Reader as Reader

# TODO: Make viability measure train data be a Cases object
if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    epochs = 50
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    ft_mode = FeatureModes.FULL
    epochs = 50
    topk = 5
    feature_len = reader.current_feature_len
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.get_loss_and_metrics()}
    
    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), tr_outcomes = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), y_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    fa_events, fa_features, fa_outcome = fa_events[y_labels[:, 0]==1][:1], fa_features[y_labels[:, 0]==1][:1], y_labels[y_labels[:, 0]==1][:1]
    training_cases = Cases(tr_events, tr_features, tr_outcomes)
    facual_cases = Cases(fa_events, fa_features, fa_outcome)

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()
    
    all_models_generators = os.listdir(PATH_MODELS_GENERATORS)
    generator:TensorflowModelMixin = tf.keras.models.load_model(PATH_MODELS_GENERATORS / all_models_generators[-1], custom_objects=custom_objects_generator)
    print("GENERATOR")
    generator.summary()
    
    cf_events, cf_features = generator.predict([np.repeat(fa_events, 10, axis=0),np.repeat(fa_features, 10, axis=0) ])    
    # cf_events, cf_features = reverse_sequence_2(cf_events), reverse_sequence_2(cf_features)
    viability = ViabilityMeasure(vocab_len, max_len, training_cases, predictor)
    
    viability_values = viability(fa_events, fa_features, cf_events, cf_features)
    print("VIABILITIES")
    print(viability_values)
    
    print("Test END-TO-END")
    evaluator = ViabilityMeasure(vocab_len, max_len, training_cases, predictor) 
    cbg_generator = CaseBasedGeneratorModel(training_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, sample_size=max(topk, 1000))
    print(case_based_generator.generate(facual_cases))

    print("Test MODULAR VIABILITY")
    evaluator = ViabilityMeasure(vocab_len, max_len, training_cases, predictor) 
    cbg_generator = CaseBasedGeneratorModel(training_cases, evaluator=evaluator, ft_mode=ft_mode, vocab_len=vocab_len, max_len=max_len, feature_len=feature_len)
    case_based_generator = CaseBasedGeneratorWrapper(predictor=predictor, generator=cbg_generator, evaluator=evaluator, topk=topk, measure_mask=MeasureMask(False, True, False, True), sample_size=max(topk, 1000))
    print(case_based_generator.generate(facual_cases))
    print("DONE")