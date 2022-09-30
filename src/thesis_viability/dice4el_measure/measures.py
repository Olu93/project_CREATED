from __future__ import annotations
import numpy as np
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
import thesis_viability.helper.base_distances as distances
from thesis_commons.representations import BetterDict, Cases
from thesis_viability.helper.base_distances import MeasureMixin
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein, SymbolicLevenshtein, SymbolicProximity
from thesis_commons.distributions import DataDistribution, GaussianParams, PFeaturesGivenActivity, TransitionProbability
from thesis_viability.outcomellh.outcomllh_measure import ImprovementMeasure

class Dice4ELProximityMeasure(MeasureMixin):
    def init(self, **kwargs) -> Dice4ELProximityMeasure:
        super().init(**kwargs)
        self.dist = SymbolicProximity(self.vocab_len, self.max_len)
        return self 
    
    
class Dice4ELSparcityMeasure(MeasureMixin):
    def init(self, **kwargs) -> Dice4ELSparcityMeasure:
        super().init(**kwargs)
        self.dist = SymbolicLevenshtein(self.vocab_len, self.max_len)
        return self    


class Dice4ELDiversityMeasure(MeasureMixin):
    def init(self, **kwargs) -> Dice4ELSparcityMeasure:
        super().init(**kwargs)
        self.dist = SymbolicProximity(self.vocab_len, self.max_len)
        return self    
    

    
    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> Dice4ELSparcityMeasure:
        with np.errstate(divide='ignore', invalid='ignore'):
            results = self.dist((*fa_cases.cases, ), (*cf_cases.cases, ))
            is_the_same = results == results.T
            ill = 1-np.mean(is_the_same, axis=-1)
            self.results = ill
        return self
    
class Dice4ELPlausibilityMeasure(DatalikelihoodMeasure):
    def init(self, **kwargs) -> Dice4ELSparcityMeasure:
        super().init(**kwargs)
        self.vault = [tuple(row) for row in self.data_distribution.events]
        return self    
    
    
    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> Dice4ELSparcityMeasure:
        results = self.custom_function((*cf_cases.cases, ))
        is_the_same = results == results.T
        ill = 1-np.mean(is_the_same, axis=-1)
        self._results = ill
        self._len_cases = len(fa_cases.events)
        return self
    
    def custom_function(self, s1):
        s1_ev, s1_ft = s1
        s1_batch_size, s1_seq_len, s1_ft_len = s1_ft.shape
        tmp = []
        for row in s1_ev:
            tmp.append(tuple(row))
        result = np.array(tmp)[..., None] 
        return result

    def normalize(self):
        super(DatalikelihoodMeasure, self).normalize()
        return self
    
class Dice4ELCategoryChangeMeasure(ImprovementMeasure):
    def init(self, **kwargs) -> Dice4ELSparcityMeasure:
        self.set_evaluator(distances.BaseDistance())
        super().init(**kwargs)
        return self    

    
    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> Dice4ELSparcityMeasure:
        factual_probs, new_cf_probs = fa_cases.outcomes, cf_cases.outcomes
        improvements = (factual_probs == new_cf_probs.T)*1
        self.llh: np.ndarray = new_cf_probs
        self.results: np.ndarray = improvements
        return self
    
    def normalize(self):
        super(ImprovementMeasure, self).normalize()
        return self