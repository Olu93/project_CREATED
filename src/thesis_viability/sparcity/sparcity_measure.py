from __future__ import annotations


import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import Cases
from thesis_viability.helper.base_distances import MeasureMixin
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein


class SparcityMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len):
        super(SparcityMeasure, self).__init__(vocab_len, max_len)
        self.dist = DamerauLevenshstein(vocab_len, max_len, distances.SparcityDistance())

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> SparcityMeasure:
        self.results = 1 / self.dist((*fa_cases.cases, ), (*cf_cases.cases, ))
        return self

    def normalize(self) -> SparcityMeasure:
        normalizing_constants = self.dist.normalizing_constants
        self.normalized_results = 1 - ((1 / self.results) / normalizing_constants)
        return self
