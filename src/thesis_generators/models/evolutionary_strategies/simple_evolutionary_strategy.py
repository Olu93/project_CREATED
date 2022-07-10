import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models

from thesis_commons import random
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.functions import extract_padding_mask
from thesis_commons.modes import (DatasetModes, FeatureModes, MutationMode, TaskModes)
from thesis_commons.representations import Cases, MutatedCases
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as Generator
from thesis_generators.models.evolutionary_strategies.evolutionary_strategy import CutPointCrossoverMixin, ElitismSelectionMixin, KPointCrossoverMixin, EvolutionaryStrategy, InitialPopulationMixin, DefaultMutationMixin, RouletteWheelSelectionMixin, TournamentSelectionMixin
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import OutcomeMockReader as Reader
from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG = True

# Tricks
# https://cs.stackexchange.com/a/54835


# TODO: Test if cf change is meaningful by test if likelihood flipped decision
# class SimpleEvolutionStrategy(InitialPopulationMixin, ElitismSelectionMixin, CutPointCrossoverMixin, DefaultMutationMixin, EvolutionaryStrategy):
#     def __init__(self, max_iter, evaluator: ViabilityMeasure, **kwargs) -> None:
#         super().__init__(max_iter=max_iter, evaluator=evaluator, **kwargs)


DEBUG = True
if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    epochs = 1000
    reader = Reader(mode=task_mode).init_meta(skip_dynamics=True)
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}
    custom_objects_generator = {obj.name: obj for obj in Generator.init_metrics()}

    # generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    take = 2
    factual_cases = Cases(fa_events[:take], fa_features[:take], fa_labels[:take, 0])

    all_models_predictors = os.listdir(PATH_MODELS_PREDICTORS)
    predictor = models.load_model(PATH_MODELS_PREDICTORS / all_models_predictors[-1], custom_objects=custom_objects_predictor)
    print("PREDICTOR")
    predictor.summary()

    viability = ViabilityMeasure(reader.vocab_len, reader.max_len, (tr_events, tr_features), predictor)

    # generator = SimpleEvolutionStrategy(
    #     evaluator=viability,
    #     vocab_len=reader.vocab_len,
    #     max_len=reader.max_len,
    #     feature_len=reader.num_event_attributes,
    #     max_iter=epochs,
    # )

    # results = generator(factual_cases, 5)
    # print("DONE")
    # print(generator.stats)
    # generator.stats.to_csv('tmp.csv', index=False, line_terminator='\n')
