
from typing import Tuple
from thesis_commons.random import random
from thesis_commons.model_commons import BaseModelMixin
from thesis_commons.representations import Cases, EvaluatedCases
from thesis_commons.statistics import StatInstance
from thesis_viability.viability.viability_function import ViabilityMeasure

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True

# TODO: Rename file name to baselines.py
class RandomGenerator(BaseModelMixin):
    def __init__(self, evaluator: ViabilityMeasure, *args, **kwargs):
        super(RandomGenerator, self).__init__(**kwargs)
        self.distance = evaluator


    def predict(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fa_case.cases
        _, max_len, feature_len = fa_ft.shape
        cf_ev = random.integers(0, self.vocab_len, size=(sample_size, max_len)).astype(float)
        cf_ft = random.standard_normal(size=(sample_size, max_len, feature_len))
        cf_cases = Cases(cf_ev, cf_ft)
        viab_values = self.distance.compute(fa_case, cf_cases)
        return EvaluatedCases(*cf_cases.cases, viab_values), {}


class SamplingBasedGenerator(BaseModelMixin):
    def __init__(self, evaluator: ViabilityMeasure, *args, **kwargs):
        super(SamplingBasedGenerator, self).__init__(**kwargs)
        self.distance = evaluator 
        self.data_distribution = evaluator.data_distribution   

    def predict(self, fc_case: Cases, **kwargs) -> EvaluatedCases:

        sample_size = kwargs.get('sample_size', 1000)
        cf_cases = self.data_distribution.sample(sample_size)
        viabilities = self.distance.compute(fc_case, cf_cases)
        generated_cases = EvaluatedCases(*cf_cases.cases, viabilities)

        return generated_cases, {**kwargs}


class CaseBasedGenerator(BaseModelMixin):
    def __init__(self, example_cases:Cases, evaluator: ViabilityMeasure, *args, **kwargs):
        super(CaseBasedGenerator, self).__init__(**kwargs)
        self.vault = example_cases
        self.examplars: Cases = None
        self.distance = evaluator

    def predict(self, fc_case: Cases, **kwargs) -> EvaluatedCases:

        sample_size = kwargs.get('sample_size', 1000)
        fa_ev, fa_ft = fc_case.cases
        cf_cases = self.sample_vault(sample_size).examplars
        viabilities = self.distance.compute(fc_case, cf_cases)
        generated_cases = EvaluatedCases(*cf_cases.cases, viabilities)

        return generated_cases, {**kwargs}


    def sample_vault(self, sample_size:int=1000):
        self.examplars = self.vault.sample(min(len(self.vault), sample_size))
        return self