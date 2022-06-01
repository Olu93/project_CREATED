import io
import pickle
from locale import normalize
from typing import Any, Callable
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


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()

    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)

    similarity_computer = SimilarityMeasure(reader.vocab_len, reader.max_len)

    bulk_distances = similarity_computer.compute_valuation(fa_events, fa_features, cf_events, cf_features)

    print(f"All results\n{bulk_distances}")
    if bulk_distances.sum() == 0:
        print("Hmm...")