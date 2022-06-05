import tensorflow as tf
# import keras.api._v2.keras as keras
# K = backend
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses as ls
from tensorflow.keras import metrics, models, optimizers, utils
from tensorflow.python.keras import backend as K

from thesis_commons.config import DEBUG_SEED, SEED_VALUE

# from keras.api._v2.keras import optimizers, layers, models, losses as ls, metrics, utils

# losses = tf.keras.losses
# optimizers = tf.keras.optimizers
# models = tf.keras.models
# metrics = tf.keras.metrics
# utils = tf.keras.utils
losses = ls

import numpy as np 
random = np.random.default_rng(SEED_VALUE) if DEBUG_SEED else np.random
    