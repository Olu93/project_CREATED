from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from thesis_commons.representations import BetterDict, ConfigurableMixin


def odds_ratio(a, b):
    return a / b


def likelihood_difference(a, b):
    return a - b


class BaseDistance():
    def __call__(self, A: NDArray, B: NDArray) -> NDArray:
        raise NotImplementedError("Needs the definition of a method")

    @property
    def MAX_VAL(self):
        return 99999999999


class LikelihoodDifference(BaseDistance):
    def __call__(self, A: NDArray, B: NDArray) -> NDArray:
        return A - B


class OddsRatio(BaseDistance):
    def __call__(self, A: NDArray, B: NDArray) -> NDArray:
        return A / B


class SparcityDistance(BaseDistance):
    def __call__(self, a, b):
        differences = a != b
        num_differences = differences.sum(axis=-1)
        return num_differences


class EuclidianDistance(BaseDistance):
    def __call__(self, A, B):
        return np.linalg.norm(A - B, axis=-1)


# https://stackoverflow.com/a/20687984/4162265
class CosineDistance(BaseDistance):
    def __call__(self, A, B):

        # squared magnitude of preference vectors (number of occurrences)
        numerator = (A * B).sum(-1)

        # inverse squared magnitude
        denominator = 1 / np.sqrt((A**2).sum(-1)) * np.sqrt((B**2).sum(-1))

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        denominator[np.isnan(denominator)] = 0

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine_similarity = numerator * denominator
        cosine_distance = 1 - cosine_similarity
        return cosine_distance


class MeasureMixin(ConfigurableMixin, ABC):
    def __init__(self) -> None:
        self.results = None
        self.normalized_results = None
        self.max_len = None
        self.vocab_len = None

    @abstractmethod
    def init(self, **kwargs) -> MeasureMixin:
        if (not self.vocab_len) or (not self.max_len):
            raise Exception(f"Configuration is missing: vocab_len={self.vocab_len} max_len={self.max_len}")
        return self

    def set_max_len(self, max_len) -> MeasureMixin:
        self.max_len = max_len
        return self

    def set_vocab_len(self, vocab_len) -> MeasureMixin:
        self.vocab_len = vocab_len
        return self

    def normalize(self):
        print("WARNING: Normalization was not implemented!")
        self.normalized_results = self.results

    def __str__(self):
        string_output = f"\nResults:\n{self.results}\nNormed Results:\n{self.normalized_results}\n"
        return string_output
    
    def get_config(self) -> BetterDict:
        return BetterDict(super().get_config()).merge({"vocab_len":self.vocab_len, "max_len":self.max_len})
