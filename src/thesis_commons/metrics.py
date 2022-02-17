from typing import List
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import losses

# TODO: Streamline Masking by using Mixin
# TODO: Think of applying masking with an external mask variable. Would elimate explicit computation.
# TODO: Streamline by adding possibility of y_pred = [y_pred, z_mean, z_log_var] possibility with Mixin
# TODO: Fix imports


class CustomLoss(keras.losses.Loss):

    def __init__(self, reduction=None, name=None, **kwargs):
        super(CustomLoss, self).__init__(reduction=reduction or keras.losses.Reduction.SUM, name=name or self.__class__.__name__, **kwargs)
        self.kwargs = kwargs

    def get_config(self):
        cfg = {**self.kwargs, **super().get_config()}
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MSpCatCE(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        y_true_pads = tf.cast(y_true, tf.int64) == 0
        y_pred_pads = tf.argmax(y_pred, axis=-1) == 0
        mask = tf.not_equal(y_true_pads, y_pred_pads)
        results = self.loss(y_true, y_pred, mask)
        return results


class MSpCatAcc(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.keras.metrics.sparse_categorical_accuracy

    def call(self, y_true, y_pred):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)

        y_true_pads = y_argmax_true == 0
        y_pred_pads = y_argmax_pred == 0
        padding_mask = tf.not_equal(y_true_pads, y_pred_pads)

        y_masked_true = tf.boolean_mask(y_argmax_true, padding_mask)
        y_masked_pred = tf.boolean_mask(y_pred, padding_mask)
        results = self.loss(y_masked_true, y_masked_pred)
        results = tf.reduce_sum(results, axis=0)
        return results


class MEditSimilarity(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.edit_distance

    def call(self, y_true, y_pred):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)

        y_true_pads = y_argmax_true == 0
        y_pred_pads = y_argmax_pred == 0
        padding_mask = tf.not_equal(y_true_pads, y_pred_pads)

        y_ragged_true = tf.ragged.boolean_mask(y_argmax_true, padding_mask)
        y_ragged_pred = tf.ragged.boolean_mask(y_argmax_pred, padding_mask)

        truth = y_ragged_true.to_sparse()
        hypothesis = y_ragged_pred.to_sparse()
        # tf.print("After conversion")
        # tf.print(tf.sparse.to_dense(truth))
        # tf.print(tf.sparse.to_dense(hypothesis))

        edit_distance = self.loss(hypothesis, truth)
        return 1 - tf.reduce_mean(edit_distance)


class JoinedLoss(CustomLoss):

    def __init__(self, losses: List[tf.keras.losses.Loss], reduction=None, name=None):
        super().__init__(reduction=reduction or keras.losses.Reduction.AUTO, name=name or f"joined_{'_'.join([l.name for l in losses])}")
        self.losses = losses

    def call(self, y_true, y_pred):
        result = 0
        for loss in self.losses:
            result += loss(y_true, y_pred)

        return result

# USELESS
class CustomMetric(keras.metrics.Metric):
    LOSS_FUNC = "l_fn"
    LOSS_VAL = "l_curr_val"

    def __init__(self, losses: List[tf.keras.losses.Loss], name=None, **kwargs):
        super(CustomMetric, self).__init__(name=name or self.__class__.__name__)
        self.all_losses = {loss.name: {CustomMetric.LOSS_VAL: self.add_weight(name=loss.name, initializer='zeros'), CustomMetric.LOSS_FUNC: loss} for loss in losses}

    def update_state(self, y_true, y_pred, sample_weight=None):
        for loss_name, loss_content in self.all_losses.items():
            new_loss = loss_content.get(CustomMetric.LOSS_FUNC)(y_true, y_pred)
            loss_content.get(CustomMetric.LOSS_VAL).assign_add(new_loss)

    def result(self):
        return {loss_name: loss_content.get(CustomMetric.LOSS_VAL) for loss_name, loss_content in self.all_losses.items()}

    def reset_states(self):
        for loss_name, loss_content in self.all_losses.items():
            loss_content.get(CustomMetric.LOSS_VAL).assign(0)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskedSpCatAcc(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(MaskedSpCatAcc, self).__init__(**kwargs)
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

    def __init__(self, reduction=keras.losses.Reduction.NONE):
        # I think it works because the reduction is done on the sample weight level and not here.
        super().__init__(reduction=reduction)
        # super().__init__()
        # self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        result = self.loss(y_true, y_pred)
        return result


class CustomSpCatAcc(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(CustomSpCatAcc, self).__init__(**kwargs)
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

    def __init__(self, **kwargs):
        super(MaskedEditSimilarity, self).__init__(**kwargs)
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


class VAELoss(keras.losses.Loss):

    def __init__(self, reduction=keras.losses.Reduction.SUM):
        super().__init__(reduction=reduction)

    def call(self, y_true, inputs):
        y_pred, z_mean, z_log_sigma = inputs
        sequence_length = y_true.shape[1]
        reconstruction = K.mean(K.square(y_true - y_pred)) * sequence_length
        kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return (reconstruction, kl)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VAEReconstructionLoss(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        reconstruction = K.mean(K.square(y_true - y_pred))
        return reconstruction


class VAEKullbackLeibnerLoss(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, z_mean, z_log_sigma):
        kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        return kl


# TODO: Streamline this by using CustomLoss and CustomMetric as Mixin
class VAEMetric(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(VAEMetric, self).__init__(**kwargs)
        self.acc_value = tf.constant(0)
        self.loss = VAELoss()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_value = self.loss(y_true, y_pred)

    def result(self):
        return self.acc_value

    def reset_states(self):
        self.acc_value = tf.constant(0)

    def get_config(self):
        return super().get_config()


def cross_entropy_function(self, y_true, y_pred):
    # prone to numerical issues
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
