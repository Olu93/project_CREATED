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

    def _to_discrete(self, y_true, y_pred):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(tf.argmax(y_pred, -1), tf.int64)
        return y_argmax_true, y_argmax_pred

    def _construct_mask(self, y_argmax_true, y_argmax_pred):
        y_true_pads = y_argmax_true != 0
        y_pred_pads = y_argmax_pred != 0
        padding_mask = keras.backend.any(keras.backend.stack([y_true_pads, y_pred_pads], axis=0), axis=0)
        return padding_mask


class MSpCatCE(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        y_argmax_true, y_argmax_pred = self._to_discrete(y_true, y_pred)
        padding_mask = self._construct_mask(y_argmax_true, y_argmax_pred)
        results = self.loss(y_true, y_pred, padding_mask)
        return results


class MSpCatAcc(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.keras.metrics.sparse_categorical_accuracy

    def call(self, y_true, y_pred):
        y_argmax_true, y_argmax_pred = self._to_discrete(y_true, y_pred)
        padding_mask = self._construct_mask(y_argmax_true, y_argmax_pred)
        y_masked_true = tf.boolean_mask(y_argmax_true, padding_mask)
        y_masked_pred = tf.boolean_mask(y_pred, padding_mask)
        results = self.loss(y_masked_true, y_masked_pred)
        results = tf.reduce_mean(results, axis=0)
        return results


class MEditSimilarity(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.edit_distance

    def call(self, y_true, y_pred):
        y_argmax_true, y_argmax_pred = self._to_discrete(y_true, y_pred)
        padding_mask = self._construct_mask(y_argmax_true, y_argmax_pred)

        y_ragged_true = tf.ragged.boolean_mask(y_argmax_true, padding_mask)
        y_ragged_pred = tf.ragged.boolean_mask(y_argmax_pred, padding_mask)

        truth = y_ragged_true.to_sparse()
        hypothesis = y_ragged_pred.to_sparse()

        edit_distance = self.loss(hypothesis, truth)
        return 1 - tf.reduce_mean(edit_distance)


class MCatEditSimilarity(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)
        self.loss = tf.edit_distance

    def call(self, y_true, y_pred):
        y_argmax_true = tf.cast(y_true, tf.int64)
        y_argmax_pred = tf.cast(y_pred, tf.int64)
        padding_mask = self._construct_mask(y_argmax_true, y_argmax_pred)

        y_ragged_true = tf.ragged.boolean_mask(y_argmax_true, padding_mask)
        y_ragged_pred = tf.ragged.boolean_mask(y_argmax_pred, padding_mask)

        truth = y_ragged_true.to_sparse()
        hypothesis = y_ragged_pred.to_sparse()

        edit_distance = self.loss(hypothesis, truth)
        return 1 - tf.reduce_mean(edit_distance)


class VAEReconstructionLoss(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        reconstruction = K.mean(K.square(y_true - y_pred), axis=-1)
        return reconstruction


class SimpleKLDivergence(CustomLoss):

    def __init__(self, reduction=None, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, z_mu, z_log_sigma):
        kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma))
        return kl


class GeneralKLDivergence(CustomLoss):

    def __init__(self, reduction=None, name=None, **kwargs):
        super().__init__(reduction, name, **kwargs)

    def call(self, dist_1, dist_2):
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        z_mean_1, z_log_sigma_1 = dist_1
        z_mean_2, z_log_sigma_2 = dist_2
        z_sigma_1 = K.exp(z_log_sigma_1)
        z_sigma_2 = K.exp(z_log_sigma_2)
        z_sigma_2_inv = (1 / z_sigma_2)
        det_1 = K.prod(z_sigma_1, axis=-1)
        det_2 = K.prod(z_sigma_2, axis=-1)
        log_det = K.log(det_2 / det_1)
        d = z_mean_1.shape[-1]
        tr_sigmas = K.sum(z_sigma_2_inv * z_sigma_1, axis=-1)
        mean_diffs = (z_mean_2 - z_mean_1)
        last_term = K.sum(mean_diffs * z_sigma_2_inv * mean_diffs, axis=-1)
        combined = 0.5 * (log_det - d + tr_sigmas + last_term)

        return combined


class JoinedLoss(CustomLoss):

    def __init__(self, losses: List[tf.keras.losses.Loss] = [], reduction=None, name=None):
        name = name if name else f"{'_'.join([l.name for l in losses])}" if losses else None
        super().__init__(reduction=reduction or keras.losses.Reduction.AUTO, name=name)
        self._losses_decomposed = {}
        if losses:
            self.losses = losses

    def call(self, y_true, y_pred):
        result = 0
        if len(self.losses) > 1:
            for loss in self.losses:
                tmp = loss(y_true, y_pred)
                result += tmp
                self._losses_decomposed[loss.name] = tmp
            self._losses_decomposed[self.name] = result
        else:
            loss = self.losses[0]
            result = loss(y_true, y_pred)
            self._losses_decomposed[loss.name] = result

        return result

    @property
    def composites(self):
        return self._losses_decomposed


class ELBOLoss(JoinedLoss):

    def __init__(self, reduction=keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss = VAEReconstructionLoss(keras.losses.Reduction.AUTO)
        self.kl_loss = SimpleKLDivergence(keras.losses.Reduction.AUTO)

    def call(self, y_true, y_pred):
        x_true = y_true
        x_rec, z_mean, z_logvar = y_pred
        rec_loss = self.rec_loss(x_true, x_rec)
        kl_loss = self.kl_loss(z_mean, z_logvar)
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss"] = rec_loss
        self._losses_decomposed["elbo_loss"] = rec_loss + kl_loss
        return rec_loss + kl_loss


class SeqELBOLoss(JoinedLoss):

    def __init__(self, reduction=keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss = VAEReconstructionLoss(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.kl_loss = GeneralKLDivergence(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        xt_true = y_true
        xt_sample, zt_transition, zt_inference = y_pred
        rec_loss = self.rec_loss(xt_true, xt_sample)
        kl_loss = self.kl_loss(zt_inference, zt_transition)
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss"] = rec_loss
        self._losses_decomposed["elbo_loss"] = rec_loss + kl_loss
        return rec_loss + kl_loss


class MetricWrapper(keras.metrics.Metric):

    def __init__(self, loss, name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.acc = tf.constant(0)
        self.loss = loss

    def update_state(self, y_true, y_pred, *args, **kwargs):
        self.acc= self.loss(y_true, y_pred)
    
    def result(self):
        return self.acc
    
    def reset_states(self):
        self.acc = tf.constant(0)
