import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import losses, metrics
# TODO: Maaaaaybe... Put into thesis_commons package


class MaskedSpCatCE(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, name="masked_spcat_ce", reduction=keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction)

    def call(self, y_true, y_pred):
        # Initiate mask matrix
        y_true_pads = tf.cast(y_true, tf.int64) == 0
        y_pred_pads = tf.argmax(y_pred, axis=-1) == 0
        mask = ~ (y_true_pads & y_pred_pads)
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
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskedSpCatAcc(tf.keras.metrics.Metric):
    def __init__(self, name="masked_spcat_acc", **kwargs):
        super(MaskedSpCatAcc, self).__init__(name=name, **kwargs)
        self.acc_value = tf.constant(0)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)

        y_true_pads = y_argmax_true == 0
        y_pred_pads = y_argmax_pred == 0
        padding_mask = ~(y_true_pads & y_pred_pads)

        y_masked_true = tf.boolean_mask(y_argmax_true, padding_mask)
        y_masked_pred = tf.boolean_mask(y_pred, padding_mask)
        self.acc_value = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_masked_true, y_masked_pred))

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomSpCatCE(keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, name="custom_spcat_ce", reduction=keras.losses.Reduction.NONE):
        # I think it works because the reduction is done on the sample weight level and not here.
        super().__init__(name=name, reduction=reduction)
        # super().__init__()
        # self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        # Initiate mask matrix

        # tf.print("weights")
        # tf.print(y_true, summarize=10)
        # tf.print("")
        # tf.print(tf.argmax(y_pred, axis=-1), summarize=10)
        result = self.loss(y_true, y_pred)
        # tf.print(self.loss.reduction)
        # result = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # result = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
        # tf.print(result)
        return result

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomSpCatAcc(tf.keras.metrics.Metric):
    def __init__(self, name="custom_spcat_accuracy", **kwargs):
        super(CustomSpCatAcc, self).__init__(name=name, **kwargs)
        self.acc_value = tf.constant(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_argmax_true = tf.cast(y_true, tf.int64)
        # y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)
        # tf.print(sample_weight)
        self.acc_value = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskedEditSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name="masked_edit_sim", **kwargs):
        super(MaskedEditSimilarity, self).__init__(name=name, **kwargs)
        self.acc_value = tf.constant(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)

        y_true_pads = y_argmax_true == 0
        y_pred_pads = y_argmax_pred == 0
        padding_mask = ~(y_true_pads & y_pred_pads)
        # tf.print("Mask")
        # tf.print(padding_mask)
        # tf.print("Inputs")
        # tf.print(y_argmax_true)
        # tf.print(y_argmax_pred)
        y_ragged_true = tf.ragged.boolean_mask(y_argmax_true, padding_mask)
        y_ragged_pred = tf.ragged.boolean_mask(y_argmax_pred, padding_mask)

        truth = y_ragged_true.to_sparse()
        hypothesis = y_ragged_pred.to_sparse()
        # tf.print("After conversion")
        # tf.print(tf.sparse.to_dense(truth))
        # tf.print(tf.sparse.to_dense(hypothesis))

        edit_distance = tf.edit_distance(hypothesis, truth)
        self.acc_value = 1 - tf.reduce_mean(edit_distance)

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


custom_metrics_default = {}
