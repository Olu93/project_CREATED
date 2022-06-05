
import numpy as np


def odds_ratio(a, b):
    return a / b


def likelihood_difference(a, b):
    return a - b


class BaseDistance():
    def __call__(self, a, b):
        raise NotImplementedError("Needs the definition of a method")

    @property
    def MAX_VAL(self):
        return 99999999999


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


class MeasureMixin():
    def __init__(self, vocab_len, max_len) -> None:
        self.results = None
        self.normalized_results = None
        self.vocab_len = vocab_len
        self.max_len = max_len

    def normalize(self):
        print("WARNING: Normalization was not implemented!")
        self.normalized_results = self.result

    def __str__(self):
        string_output = f"\nResults:\n{self.results}\nNormed Results:\n{self.normalized_results}\n"
        return string_output
