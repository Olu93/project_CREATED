import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import losses, metrics


class VAELoss(keras.losses.Loss):

    def __init__(self, name="vae_loss", reduction=keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, inputs):
        y_pred, z_mean, z_log_sigma = inputs
        sequence_length = len(y_true[0])
        reconstruction = tf.keras.mean(tf.keras.square(y_true - y_pred)) * sequence_length
        kl = -0.5 * tf.keras.mean(1 + z_log_sigma - tf.keras.square(z_mean) - tf.keras.exp(z_log_sigma))
        return reconstruction + kl

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)