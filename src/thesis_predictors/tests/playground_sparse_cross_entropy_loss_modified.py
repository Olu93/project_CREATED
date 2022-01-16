# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.utils import losses_utils

y_true = tf.constant([[1, 2, 1, 0, 0], [1, 2, 1, 1, 0], [1, 2, 1, 1, 2], [1, 2, 0, 0, 0]], dtype=tf.float32)
y_pred = tf.constant(
    [
        [
            [0.04, 0.95, 0.01],
            [0.10, 0.10, 0.80],
            [0.20, 0.70, 0.10],
            [0.20, 0.10, 0.70],
            [0.90, 0.00, 0.10],
        ],
        [
            [0.04, 0.01, 0.95],
            [0.10, 0.10, 0.80],
            [0.95, 0.03, 0.02],
            [0.90, 0.05, 0.05],
            [0.90, 0.00, 0.10],
        ],
        [
            [0.95, 0.04, 0.01],
            [0.10, 0.80, 0.10],
            [0.95, 0.03, 0.02],
            [0.90, 0.05, 0.05],
            [0.90, 0.00, 0.10],
        ],
        [
            [0.95, 0.04, 0.01],
            [0.10, 0.80, 0.10],
            [0.05, 0.93, 0.02],
            [0.90, 0.05, 0.05],
            [0.90, 0.00, 0.10],
        ],
    ],
    dtype=tf.float32,
)
print(y_true.shape)
print(y_pred.shape)

# %%
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.AUTO)
print(cce(y_true, y_pred).numpy())
print(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))


# %%
class SparseCrossEntropyLoss(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, reduction=keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        y_true_end = tf.argmax(tf.cast(tf.equal(y_true, 0), tf.float32), axis=-1)
        y_pred_end = tf.argmax(tf.equal(tf.argmax(y_pred, axis=-1), 0), axis=-1)
        longest_sequence = tf.reduce_max([y_true_end, y_pred_end], axis=0)
        # Initiate mask matrix
        weights = tf.zeros_like(y_true)
        # Craft mask indices with fix in case longest sequence is 0
        # bsize = tf.range(len(longest_sequence))
        tmp = []
        for num_idx in longest_sequence:
            tmp.append(tf.concat([tf.ones(num_idx), tf.zeros(weights.shape[1] - num_idx)], axis=-1))
        tmp2 = tf.stack(tmp)
        result = self.loss(y_true, y_pred, weights)
        return result


# class SparseCrossEntropyLoss(keras.losses.Loss):
#     """
#     Args:
#       reduction: Type of tf.keras.losses.Reduction to apply to loss.
#       name: Name of the loss function.
#     """
#     def __init__(self, reduction=keras.losses.Reduction.AUTO):
#         super().__init__(reduction=reduction)
#         self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

#     def call(self, y_true, y_pred):
#         y_true_end = tf.argmax(tf.cast(tf.equal(y_true, 0), tf.float32), axis=-1)
#         y_pred_end = tf.argmax(tf.equal(tf.argmax(y_pred, axis=-1), 0), axis=-1)
#         longest_sequence = tf.reduce_max([y_true_end, y_pred_end], axis=0)
#         # Initiate mask matrix
#         weights = tf.zeros_like(y_true)
#         # Craft mask indices with fix in case longest sequence is 0
#         bsize = tf.range(len(longest_sequence))
#         weight_indices = [(idx,col) for idx in bsize for col in tf.range(longest_sequence[idx] if longest_sequence[idx]!=0 else y_true.shape[1])]
#         weight_indices = list(zip(*weight_indices))
#         weight_indices = np.array(weight_indices).T
#         # Set masks
#         # weights[[*weight_indices.T]] = 1
#         weights = tf.sparse.to_dense(tf.SparseTensor(weight_indices, [1]*len(weight_indices), weights.shape))

#         result = self.loss(y_true, y_pred, weights)
#         return result

# class SparseCrossEntropyLoss(keras.losses.Loss):
#     """
#     Args:
#       reduction: Type of tf.keras.losses.Reduction to apply to loss.
#       name: Name of the loss function.
#     """
#     def __init__(self, reduction=keras.losses.Reduction.AUTO):
#         super().__init__(reduction=reduction)
#         self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

#     def call(self, y_true, y_pred):
#         y_true_end = tf.argmax(tf.cast(tf.equal(y_true, 0), tf.float32), axis=-1)
#         y_pred_end = tf.argmax(tf.equal(tf.argmax(y_pred, axis=-1), 0), axis=-1)
#         longest_sequence = tf.reduce_max([y_true_end, y_pred_end], axis=0)
#         # Initiate mask matrix
#         weights = tf.zeros_like(y_true)
#         # Craft mask indices with fix in case longest sequence is 0
#         weight_indices = [(idx,col) for idx, num_seq in enumerate(longest_sequence) for col in tf.range(num_seq if num_seq!=0 else y_true.shape[1])]
#         weight_indices = list(zip(*weight_indices))
#         weight_indices = np.array(weight_indices).T
#         # Set masks
#         # weights[[*weight_indices.T]] = 1
#         weights = tf.sparse.to_dense(tf.SparseTensor(weight_indices, [1]*len(weight_indices), weights.shape))

#         result = self.loss(y_true, y_pred, weights)
#         return result

# img = np.zeros((4, 5))
# coords = np.array([[0,0],[0, 1], [1, 2], [2, 3], [3, 4], [3, 0], [3, 1],[3, 2]])
# vals = np.array([1, 2, 3, 4])
# img[[*coords.T]] = 255
# print(img)
cce = SparseCrossEntropyLoss()
cce(tf.constant(y_true), tf.constant(y_pred))

# %%
