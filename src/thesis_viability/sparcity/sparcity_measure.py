from __future__ import annotations

import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin
from thesis_viability.helper.base_distances import MeasureMixin
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein
import numpy as np

class SparcityMeasure(MeasureMixin):
    def init(self, **kwargs) -> SparcityMeasure:
        super().init(**kwargs)
        self.dist = DamerauLevenshstein(self.vocab_len, self.max_len, distances.SparcityDistance())
        return self

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> SparcityMeasure:
        with np.errstate(divide='ignore', invalid='ignore'):
            self.results = 1 / self.dist((*fa_cases.cases, ), (*cf_cases.cases, ))
        return self

    def normalize(self) -> SparcityMeasure:
        normalizing_constants = self.dist.normalizing_constants
        # This is to briefly re-revert the similarity to a distance and then normalise using the constants
        # Having a distance between 0 and 1 allows to now just compute 1-dist = similarity
        self.normalized_results = 1 - ((1 / self.results) / normalizing_constants)
        return self

    def get_config(self) -> BetterDict:
        return super().get_config().merge({"type":type(self).__name__})