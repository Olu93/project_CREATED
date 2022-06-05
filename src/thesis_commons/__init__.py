import random as rng

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
