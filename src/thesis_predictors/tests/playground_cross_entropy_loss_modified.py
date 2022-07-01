# %%
import tensorflow as tf
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models
import tensorflow.keras as keras
from helper.loss_functions import CrossEntropyLoss
from tensorflow.keras import layers
from tensorflow.keras.utils import losses_utils

y_true = tf.constant([
    [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
    # [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
    # [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
], dtype=tf.float32)
y_pred = tf.constant([
    [[0.05, 0.95, 0.00], [0.10, 0.80, 0.10], [0.20, 0.70, 0.10], [0.20, 0.01, 0.07]],
    # [[1.00, 0.00, 0.00], [0.10, 0.80, 0.10], [0.10, 0.80, 0.01], [0.01, 0.01, 0.98]],
    # [[0.95, 0.00, 0.05], [0.10, 0.90, 0.00], [0.05, 0.90, 0.05], [0.02, 0.96, 0.02]],
], dtype=tf.float32)

# %%
cce = losses.CategoricalCrossentropy(reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)
cce(y_true, y_pred).numpy()
# %%
results = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1).numpy()
tensor_without_nans = tf.reduce_sum(tf.where(tf.math.is_nan(results), tf.zeros_like(results), results))
tensor_without_nans.numpy()
# %%


class CrossEntropyLossModified(CrossEntropyLoss):
    def __init__(self, reduction=losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.offset = 2
        self.pad = layers.ZeroPadding1D((0, self.offset))

    def call(self, y_true, y_pred):
        y_true = y_true[:, self.offset:]
        y_true = self.pad(y_true)
        print(y_pred)
        print(y_true)
        result = self.loss(y_true, y_pred)
        return result


cce = CrossEntropyLossModified()
cce(y_true, y_pred).numpy()

# %%
