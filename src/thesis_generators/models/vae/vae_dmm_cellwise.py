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
from tensorflow.python.ops import array_ops


class DMMModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        # self.feature_len = embed_dim + self.feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = Sampler(self.ff_dim)
        self.emitter_events = EmissionModel(embed_dim)
        self.emitter_features = EmissionModel(self.feature_len)
        self.masker = layers.Masking()

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        sampled_z_tra_mean_list = []
        sampled_z_tra_logvar_list = []
        sampled_z_inf_mean_list = []
        sampled_z_inf_logvar_list = []
        sampled_x_emi_mean_list_events = []
        sampled_x_emi_logvar_list_events = []
        sampled_x_emi_mean_list_features = []
        sampled_x_emi_logvar_list_features = []
        gt_backwards = self.future_encoder(inputs)

        zt_sample = tf.keras.backend.zeros_like(gt_backwards)[:, 0]
        # zt_sample = tf.repeat(self.initial_z, len(gt_backwards), axis=0)
        for t in range(inputs.shape[1]):
            zt_prev = zt_sample
            xt = inputs[:, t]
            gt = gt_backwards[:, t]

            z_transition_mu, z_transition_logvar = self.state_transitioner(zt_prev)
            z_inf_mu, z_inf_logvar = self.inferencer([gt, zt_prev])

            zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
            xt_emi_mu_events, xt_emi_logvar_events = self.emitter_events(zt_sample)
            xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample)

            sampled_z_tra_mean_list.append(z_transition_mu)
            sampled_z_tra_logvar_list.append(z_transition_logvar)
            sampled_z_inf_mean_list.append(z_inf_mu)
            sampled_z_inf_logvar_list.append(z_inf_logvar)
            sampled_x_emi_mean_list_events.append(xt_emi_mu_events)
            sampled_x_emi_logvar_list_events.append(xt_emi_logvar_events)
            sampled_x_emi_mean_list_features.append(xt_emi_mu_features)
            sampled_x_emi_logvar_list_features.append(xt_emi_logvar_features)

        sampled_z_tra_mean = tf.stack(sampled_z_tra_mean_list, axis=1)
        sampled_z_tra_logvar = tf.stack(sampled_z_tra_logvar_list, axis=1)
        sampled_z_inf_mean = tf.stack(sampled_z_inf_mean_list, axis=1)
        sampled_z_inf_logvar = tf.stack(sampled_z_inf_logvar_list, axis=1)
        sampled_x_emi_mean_events = tf.stack(sampled_x_emi_mean_list_events, axis=1)
        sampled_x_emi_logvar_events = tf.stack(sampled_x_emi_logvar_list_events, axis=1)
        sampled_x_emi_mean_features = tf.stack(sampled_x_emi_mean_list_features, axis=1)
        sampled_x_emi_logvar_features = tf.stack(sampled_x_emi_logvar_list_features, axis=1)
        if tf.math.is_nan(K.sum(sampled_x_emi_mean_events)):
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(K.sum(sampled_x_emi_mean_events)))}")
        return [sampled_x_emi_mean_events, sampled_x_emi_logvar_events], [sampled_x_emi_mean_features,
                                                                          sampled_x_emi_logvar_features], [sampled_z_tra_mean,
                                                                                                           sampled_z_tra_logvar], [sampled_z_inf_mean, sampled_z_inf_logvar]


# https://stackoverflow.com/questions/54231440/define-custom-lstm-cell-in-keras
class CustomDynamicVAECell(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units
        super(CustomDynamicVAECell, self).__init__(**kwargs)
        self.future_block_g = layers.LSTMCell(self.units)

        self.transition_block_mu = layers.LSTMCell(self.units)
        self.transition_block_sigmasq = layers.LSTMCell(self.units)
        
        self.inference_block_mu = layers.LSTMCell(self.units)
        self.inference_block_sigmasq = layers.LSTMCell(self.units)
        
        self.emission_block_mu = layers.LSTMCell(self.units)
        self.emission_block_sigmasq = layers.LSTMCell(self.units)
        

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.future_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
        self.future_recurrent = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='recurrent_kernel')

        self.w_transition_mu, self.w_transition_logsigma, self.w_transition_rec = self.build_weights(input_shape)
        self.w_inference_mu, self.w_inference_logsigma, self.w_inference_rec = self.build_weights(input_shape)
        self.w_emission_mu, self.w_emission_logsigma, self.w_emission_rec = self.build_weights(input_shape)
        
        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def build_weights(self, input_shape, initializer='uniform', name='kernel'):
        mu = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer=initializer,
                                        name=f'{name}_mu')
        logsigma = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer=initializer,
                                        name=f'{name}_logsigma')
        recurrent = self.add_weight(shape=(input_shape[-1], self.units*2),
                                        initializer=initializer,
                                        name=f'{name}_logsigma')
        return mu, logsigma, recurrent

    def call(self, inputs, states):
        prev_future,prev_transition,prev_inference, prev_emission = states
        curr_future,curr_transition_mu,curr_inference, curr_emission = inputs
        h_future = K.dot(curr_future, self.future_kernel)
        output_future = h_future * K.dot(prev_future, self.future_recurrent)
        h_transition_mu = K.dot(curr_transition_mu, self.w_transition_mu)
        return [output_future], [output_future]
    
# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(Model):

    def __init__(self, ff_dim):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, go_backwards=True)
        self.combiner = layers.Concatenate()

    def call(self, inputs):
        x = inputs
        x = self.lstm_layer(x)
        # g_t_backwards = self.combiner([h, c])
        return x


# https://youtu.be/rz76gYgxySo?t=1450
class TransitionModel(Model):

    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_tra_mean", activation='tanh')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_tra_logvar", activation='tanh')

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
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_inf_mean", activation='tanh')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_inf_logvar", activation='tanh')

    def call(self, inputs, training=None, mask=None):
        g_t_backwards, z_t_minus_1 = inputs
        combined_input = self.combiner([g_t_backwards, z_t_minus_1])
        z_mean = self.latent_vector_z_mean(combined_input)
        z_log_var = self.latent_vector_z_log_var(combined_input)
        return z_mean, z_log_var


class Sampler(layers.Layer):
    # TODO: centralise this layer for broad use to reduce code repetition
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # Explained here https://jaketae.github.io/study/vae/
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EmissionModel(Model):

    def __init__(self, feature_len):
        super(EmissionModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(feature_len, name="z_emi_mean", activation='tanh')
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="z_emi_logvar", activation='tanh')

    def call(self, inputs):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var
