import tensorflow as tf
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        # Why log(x) - https://stats.stackexchange.com/a/486161
        z_mean, z_log_var = inputs
        # Why log(variance) - https://stats.stackexchange.com/a/486205

        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# class RandomPicker(layers.Layer):
#     def call(self, inputs):
#         z_mean, z_log_var = inputs

#         epsilon = K.random_normal(shape=tf.shape(z_mean))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon