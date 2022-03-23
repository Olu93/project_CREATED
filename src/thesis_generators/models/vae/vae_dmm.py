from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_commons import metric
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin
import thesis_generators.models.model_commons as commons
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType
import tensorflow.keras.backend as K


class DMMModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        self.emb_len = embed_dim + self.feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.emb_len)
        self.state_transitioner = TransitionModel(self.emb_len)
        self.inferencer = InferenceModel(self.emb_len)
        self.sampler = Sampler(self.ff_dim)
        self.emitter_events = EmissionModel(embed_dim)
        self.emitter_features = EmissionModel(self.feature_len)
        self.masker = layers.Masking()

    # def build(self, input_shape):
    #     self.zt_init_sample = tf.keras.backend.zeros(input_shape)
    #     # return super().build(input_shape)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):

        gt_backwards = self.future_encoder(inputs, training, mask)
        zt_sample = self.sampler([tf.zeros_like(gt_backwards), tf.zeros_like(gt_backwards)])
        z_transition_mu, z_transition_logvar = self.state_transitioner(zt_sample, training, mask)
        z_inf_mu, z_inf_logvar = self.inferencer([gt_backwards, zt_sample], training, mask)
        zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
        xt_emi_mu_events, xt_emi_logvar_events = self.emitter_events(zt_sample, training, mask)
        xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample, training, mask)

        r_tra_params = self.stack([z_transition_mu, z_transition_logvar], axis=-2)
        r_inf_params = self.stack([z_inf_mu, z_inf_logvar], axis=-2)
        r_emi_ev_params = self.stack([xt_emi_mu_events, xt_emi_logvar_events], axis=-2)
        r_emi_ft_params = self.stack([xt_emi_mu_features, xt_emi_logvar_features], axis=-2)
        return r_tra_params, r_inf_params, r_emi_ev_params, r_emi_ft_params


class Sampler(layers.Layer):
    # TODO: centralise this layer for broad use to reduce code repetition
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seqlen = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, seqlen, dim))
        # Explained here https://jaketae.github.io/study/vae/
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(Model):

    def __init__(self, ff_dim):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, go_backwards=True)
        # self.combiner = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.lstm_layer(x, training=training, mask=mask)
        # g_t_backwards = self.combiner([h, c])
        return x


# https://youtu.be/rz76gYgxySo?t=1450
class TransitionModel(Model):

    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.latent_vector_z_mean = layers.LSTM(ff_dim, name="z_tra_mean", activation='tanh', return_sequences=True)
        self.latent_vector_z_log_var = layers.LSTM(ff_dim, name="z_tra_logvar", activation='tanh', return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        z_t_minus_1 = inputs
        z_mean = self.latent_vector_z_mean(z_t_minus_1, training=training, mask=mask)
        z_log_var = self.latent_vector_z_log_var(z_t_minus_1, training=training, mask=mask)
        return z_mean, z_log_var


# https://youtu.be/rz76gYgxySo?t=1483
class InferenceModel(Model):

    def __init__(self, ff_dim):
        super(InferenceModel, self).__init__()
        self.combiner = layers.Concatenate()
        self.latent_vector_z_mean = layers.LSTM(ff_dim, name="z_inf_mean", activation='tanh', return_sequences=True)
        self.latent_vector_z_log_var = layers.LSTM(ff_dim, name="z_inf_logvar", activation='tanh', return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        g_t_backwards, z_t_minus_1 = inputs
        combined_input = self.combiner([g_t_backwards, z_t_minus_1])
        z_mean = self.latent_vector_z_mean(combined_input, training=training, mask=mask)
        z_log_var = self.latent_vector_z_log_var(combined_input, training=training, mask=mask)
        return z_mean, z_log_var





class EmissionModel(Model):

    def __init__(self, feature_len):
        super(EmissionModel, self).__init__()
        self.latent_vector_z_mean = layers.LSTM(feature_len, name="z_emi_mean", activation='tanh', return_sequences=True)
        self.latent_vector_z_log_var = layers.LSTM(feature_len, name="z_emi_logvar", activation='tanh', return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample, training=training, mask=mask)
        z_log_var = self.latent_vector_z_log_var(z_sample, training=training, mask=mask)
        return z_mean, z_log_var
