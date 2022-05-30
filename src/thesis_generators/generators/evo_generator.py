from thesis_commons.model_commons import BaseModelMixin
from thesis_generators.models.evolutionary_strategies.simple_evolutionary_strategy import SimpleEvolutionStrategy
from thesis_commons.representations import GeneratorResult
from thesis_commons.representations import Cases
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.model_commons import GeneratorMixin
import numpy as np


class SimpleEvoGenerator(GeneratorMixin):
    def __init__(self, generator: BaseModelMixin, evaluator: ViabilityMeasure, **kwargs) -> None:
        super().__init__(evaluator)
        self.generator: SimpleEvolutionStrategy = generator

    def generate(self, fa_cases: Cases) -> GeneratorResult:
        fa_events, fa_features = fa_cases.items()
        return self.generator.predict([fa_events, fa_features], fa_labels)
