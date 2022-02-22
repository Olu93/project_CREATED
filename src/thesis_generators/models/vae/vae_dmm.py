from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin
import thesis_generators.models.model_commons as commons
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType


class DMMModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, layer_dims=[13, 8, 5], *args, **kwargs):
        print(__class__)
        super(DMMModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        layer_dims = [kwargs.get("feature_len") + kwargs.get("embed_dim")] + layer_dims
        self.initial_z = tf.zeros((None, ff_dim))
        self.is_initial = True
        self.encoder_layer_dims = layer_dims
        self.decoder_layer_dims = reversed(layer_dims)
        self.combiner = layers.Concatenate()
        self.future_encoder = FutureSeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(layer_dims[0], self.max_len, self.ff_dim, self.decoder_layer_dims)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        z_t_minus_1_mean, z_t_minus_1_var = self.state_transitioner(x)
        z_transition_sampler = self.sampler([z_t_minus_1_mean, z_t_minus_1_var])
        g_t_backwards = self.future_encoder([z_t_minus_1_mean, z_t_minus_1_var])
        
        z_t_sample_minus_1 = self.sampler([g_t_backwards, z_transition_sampler])
        z_t_mean, z_t_log_var = self.decoder(z_t_sample_minus_1)
        return z_t_mean, z_t_log_var


class FutureSeqEncoder(Model):

    def __init__(self, ff_dim, layer_dims):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_state=True, return_sequences=True)
        self.combiner = layers.Concatenate()

    def call(self, inputs):
        x = inputs
        x, h, c = self.lstm_layer(x)
        g_t_backwards = self.combiner([h, c])
        return g_t_backwards


# https://youtu.be/rz76gYgxySo?t=1450
class TransitionModel(Model):

    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        z_t_minus_1 = inputs
        z_mean = self.latent_vector_z_mean(z_t_minus_1)
        z_log_var = self.latent_vector_z_log_var(z_t_minus_1)
        return z_mean, z_log_var


# https://youtu.be/rz76gYgxySo?t=1483
class InferenceModel(Model):

    def __init__(self, ff_dim):
        super(InferenceModel, self).__init__()
        self.combiner = layers.Concatenate()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        g_t_backwards, z_t_minus_1 = inputs
        combined_input = self.combiner([g_t_backwards, z_t_minus_1])
        z_mean = self.latent_vector_z_mean(combined_input)
        z_log_var = self.latent_vector_z_log_var(combined_input)
        return z_mean, z_log_var


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # TODO: Maybe remove the 0.5 and include proper log handling
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SeqDecoder(Model):
    def __init__(self, in_dim, max_len, ff_dim, layer_dims):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True)
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")
        
    def call(self, inputs):
        z_sample = inputs

        x = self.lstm_layer(z_sample)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)        
        return z_mean, z_log_var