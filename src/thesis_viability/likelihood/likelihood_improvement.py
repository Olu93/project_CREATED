import io
from math import isnan
from typing import Any, Callable
from unicodedata import is_normalized
import numpy as np
import thesis_viability.helper.base_distances as distances
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance
import scipy.stats as stats
import tensorflow as tf
import pickle
from collections import Counter
import pandas as pd

DEBUG = True

def odds_ratio(factual_likelihood, counterfactual_likelihood):
    return counterfactual_likelihood / factual_likelihood

def likelihood_difference(factual_likelihood, counterfactual_likelihood):
    return counterfactual_likelihood - factual_likelihood

class ImprovementCalculator():
    def __init__(self, prediction_model: tf.keras.Model, valuation_function: Callable) -> None:
        self.predictor = prediction_model
        self.valuator = valuation_function

    def compute_valuation(self, factual_events, factual_features, counterfactual_events, counterfactual_features):
        factual_likelihoods = self.predictor(factual_events, factual_features)
        counterfactual_likelihoods = self.predictor(counterfactual_events, counterfactual_features)
        improvements = self.valuator(factual_likelihoods.prod(-1), counterfactual_likelihoods.prod(-1))
        return improvements

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    # generative_reader = GenerativeDataset(reader)
    (events, ev_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    metric = ImprovementCalculator(events, ev_features)
    print(metric.compute_values(events, ev_features))

