import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class MaskedSpCatCE(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, balance=False, reduction=keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # Initiate mask matrix
        y_true_pads = tf.cast(y_true, tf.int64) == 0
        y_pred_pads = tf.argmax(y_pred, axis=-1) == 0
        mask = not (y_true_pads & y_pred_pads)
        # Craft mask indices with fix in case longest sequence is 0
        # tf.print("weights")
        # tf.print(y_true, summarize=10)
        # tf.print("")
        # tf.print(tf.argmax(y_pred, axis=-1), summarize=10)
        # tf.print("")
        # tf.print(mask, summarize=10)
        result = self.loss(y_true, y_pred, mask)
        return result

    def get_config(self):
        return {"reduction": self.reduction}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskedSpCatAcc(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(MaskedSpCatAcc, self).__init__(**kwargs)
        self.acc_value = tf.constant(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true[0], tf.int32)
        # y_pred = tf.cast(y_pred, tf.int32)
        self.acc_value = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true[0], y_pred))

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EditSimilarity(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(EditSimilarity, self).__init__(**kwargs)
        self.acc_value = tf.constant(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true_pads = tf.cast(y_true, tf.int64) == 0
        y_pred_pads = tf.argmax(y_pred, axis=-1) == 0
        mask = not (y_true_pads & y_pred_pads)
        
        hypothesis = tf.cast(tf.argmax(y_pred, -1), tf.int64)
        truth =  tf.cast(y_true, tf.int64)
        edit_distance = tf.edit_distance(tf.sparse.from_dense(hypothesis),tf.sparse.from_dense(truth))
        self.acc_value = 1-tf.reduce_mean(edit_distance)

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def cross_entropy_function(self, y_true, y_pred):
    # prone to numerical issues
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)