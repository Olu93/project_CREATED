from unicodedata import is_normalized

import numpy as np
import tensorflow as tf
from scipy.spatial import distance

import thesis_viability.helper.base_distances as distances
from thesis_commons.functions import stack_data
from thesis_commons.modes import (DatasetModes, FeatureModes, GeneratorModes,
                                  TaskModes)
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_readers import MockReader as Reader
from thesis_viability.helper.base_distances import MeasureMixin
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein


class SimilarityMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len) -> None:
        super(SimilarityMeasure, self).__init__(vocab_len, max_len)
        self.dist = DamerauLevenshstein(vocab_len, max_len, distances.EuclidianDistance())

    def compute_valuation(self, fa_events, fa_features, cf_events, cf_features):
        self.results = 1 / self.dist((fa_events, fa_features), (cf_events, cf_features))
        return self

    def normalize(self):
        normalizing_constants = self.dist.normalizing_constants
        self.normalized_results = 1 - ((1 / self.results) / normalizing_constants)
        return self


