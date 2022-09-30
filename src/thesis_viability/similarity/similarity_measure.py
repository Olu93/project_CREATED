from __future__ import annotations
import numpy as np
import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import BetterDict, Cases
from thesis_viability.helper.base_distances import MeasureMixin
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein, SymbolicLevenshtein, SymbolicProximity


class SimilarityMeasure(MeasureMixin):
    def init(self, **kwargs) -> SimilarityMeasure:
        super().init(**kwargs)
        self.dist = DamerauLevenshstein(self.vocab_len, self.max_len, distances.EuclidianDistance())
        return self

    def compute_valuation(self, fa_cases:Cases, cf_cases:Cases) -> SimilarityMeasure:
        with np.errstate(divide='ignore', invalid='ignore'):
            self.results = 1 / self.dist((*fa_cases.cases,), (*cf_cases.cases,))
        return self

    def normalize(self) -> SimilarityMeasure:
        normalizing_constants = self.dist.normalizing_constants
        # This is to briefly re-revert the similarity to a distance and then normalise using the constants
        # Having a distance between 0 and 1 allows to now just compute 1-dist = similarity
        self.normalized_results = 1 - ((1 / self.results) / normalizing_constants)
        return self

    def get_config(self) -> BetterDict:
        return super().get_config().merge({"type":type(self).__name__})


