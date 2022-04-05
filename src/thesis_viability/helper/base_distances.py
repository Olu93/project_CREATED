import io
from typing import Any, Callable
from unicodedata import is_normalized
import numpy as np
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance
import tensorflow as tf
import pickle


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

    def __call__(self, a, b):
        return np.linalg.norm(a - b)


class CosineDistance(BaseDistance):

    def __call__(self, a, b):

        # base similarity matrix (all dot products)
        # replace this with A.dot(A.T).toarray() for sparse representation
        similarity = np.dot(a, b.T)

        # squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)

        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)
            
        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        return cosine
