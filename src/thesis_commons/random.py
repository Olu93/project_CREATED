import random as rng
from tkinter import NONE
from numpy.typing import NDArray
import numpy as np
from numpy import random

from thesis_commons.config import DEBUG_SEED, SEED_VALUE

if DEBUG_SEED:
    rng.seed(42)
    np.random.seed(SEED_VALUE)
    print(f"Random Seed is {SEED_VALUE}")
    random = random.default_rng(SEED_VALUE)
else:
    print(f"Random Seed is not set")
    random = random.default_rng(NONE)
    
def matrix_sample(p:NDArray):
    c = p.cumsum(axis=-1)
    u = random.random((len(c), 1))
    chosen = (u < c).argmax(axis=-1, keepdims=True)
    return chosen