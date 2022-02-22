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

    # def call(self, inputs, training=None, mask=None):
    #     x = inputs
    #     z_t_minus_1_mean, z_t_minus_1_var = self.state_transitioner(x)
    #     z_transition_sampler = self.sampler([z_t_minus_1_mean, z_t_minus_1_var])
    #     g_t_backwards = self.future_encoder([z_t_minus_1_mean, z_t_minus_1_var])
        
    #     z_t_sample_minus_1 = self.sampler([g_t_backwards, z_transition_sampler])
    #     z_t_mean, z_t_log_var = self.decoder(z_t_sample_minus_1)
    #     return z_t_mean, z_t_log_var


class DMMModel(Model):

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
        self.embedder = commons.HybridEmbedderLayer(*args, **kwargs)
        # self.combiner = layers.Concatenate()
        self.future_encoder = FutureSeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.sampler = Sampler(self.encoder_layer_dims[-1])


    def train_step(self, data):
        (events, features), _ = data
        metrics_collector = {}
        # Train Process
        x_pred = []
        with tf.GradientTape() as tape:
            x = self.embedder([events, features])
            z_t_minus_1_mean, z_t_minus_1_var = self.state_transitioner(self.initial_z)
            g_t_backwards = self.future_encoder([z_t_minus_1_mean, z_t_minus_1_var])
            mu_z_t_minus_1, log_sigma_z_t_minus_1 = self.inferencer([g_t_backwards, self.initial_z])
            z_t_sample = self.sampler([mu_z_t_minus_1, log_sigma_z_t_minus_1])
            z_t_mean, z_t_log_var = self.decoder(z_t_sample)
            x_t = self.sampler([z_t_mean, z_t_log_var])
            
            
        # metrics_collector.update({m.name: m.result() for m in self.metrics})
        return metrics_collector

class DMMCell(layers.AbstractRNNCell):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.ff_dim = kwargs.get("ff_dim")
        self.vocab_len = kwargs.get("vocab_len")
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inference_encoder = InferenceEncoderModel(self.ff_dim)        
        self.sampler = Sampler(self.ff_dim)
        self.inference_decoder = InferenceDecoderModel(self.ff_dim)
        self.reconstuctor = ReconstructionModel()
        self.outputer = layers.Softmax()

    
    def call(self, inputs, states):
        gt =  inputs
        zt_minus_1 = states[0] 
        z_transition_mu, z_transition_logvar = self.state_transitioner(zt_minus_1)
        z_inference_mu, z_inference_logvar = self.inference_encoder([gt, zt_minus_1])
        zt_sample = self.sampler([z_inference_mu, z_inference_logvar])
        zt_mean, zt_logvar = self.inference_decoder(zt_sample)
        xt_sample = self.reconstuctor([zt_mean, zt_logvar])
        next_states = [zt_sample, z_transition_mu, z_transition_logvar, z_inference_mu, z_inference_logvar, zt_mean, zt_logvar]
        return super().call(xt_sample, next_states)
        
# https://youtu.be/rz76gYgxySo?t=1383        
class FutureSeqEncoder(Model):

    def __init__(self, ff_dim, layer_dims):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_state=True, return_sequences=True, go_backwards=True)
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
class InferenceEncoderModel(Model):

    def __init__(self, ff_dim):
        super(InferenceEncoderModel, self).__init__()
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

class InferenceDecoderModel(Model):
    def __init__(self, ff_dim):
        super(InferenceDecoderModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")
        
    def call(self, inputs):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)        
        return z_mean, z_log_var

class ReconstructionModel(Model):
    def __init__(self):
        super(ReconstructionModel, self).__init__()
        self.sampler = Sampler()
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        x_reconstructed = self.sampler([z_mean, z_log_var])
        return x_reconstructed    
