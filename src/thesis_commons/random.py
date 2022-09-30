import random as rng
import numpy as np
from numpy import random


DEBUG_SEED = False
SEED_VALUE = 42

if DEBUG_SEED:
    rng.seed(42)
    np.random.seed(SEED_VALUE)
    print(f"Random Seed is {SEED_VALUE}")
    random = random.default_rng(SEED_VALUE)
else:
    print(f"Random Seed is not set")
    random = random.default_rng(None)
    
def matrix_sample(p:np.ndarray):
    c = p.cumsum(axis=-1)
    u = random.random((len(c), 1))
    chosen = np.array(u < c).argmax(axis=-1, keepdims=True)
    return chosen