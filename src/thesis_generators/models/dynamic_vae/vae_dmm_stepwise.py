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



class DMMModelStepwise(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModelStepwise, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        # self.feature_len = embed_dim + self.feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = commons.Sampler()
        self.emitter_events = EmissionEvModel(self.vocab_len)
        self.emitter_features = EmissionFtModel(self.feature_len)
        self.combiner = layers.Concatenate()
        self.masker = layers.Masking()



    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)


    def call(self, inputs, training=None, mask=None):
        sampled_z_tra_mean_list = []
        sampled_z_tra_logvar_list = []
        sampled_z_inf_mean_list = []
        sampled_z_inf_logvar_list = []
        sampled_x_probs_list_events = []
        sampled_x_emi_mean_list_features = []
        sampled_x_emi_logvar_list_features = []
        x = self.future_encoder(inputs)

        zt_sample = tf.keras.backend.zeros_like(x)[:, 0]
        # zt_sample = tf.repeat(self.initial_z, len(gt_backwards), axis=0)
        for t in range(inputs.shape[1]):
            zt_prev = zt_sample
            xt = x[:, t]

            z_transition_mu, z_transition_logvar = self.state_transitioner(zt_prev)
            z_inf_mu, z_inf_logvar = self.inferencer(self.combiner([xt, zt_prev]))

            zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
            xt_emi_ev_probs = self.emitter_events(zt_sample)
            xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample)
            
            sampled_z_tra_mean_list.append(z_transition_mu)
            sampled_z_tra_logvar_list.append(z_transition_logvar)
            sampled_z_inf_mean_list.append(z_inf_mu)
            sampled_z_inf_logvar_list.append(z_inf_logvar)
            sampled_x_probs_list_events.append(xt_emi_ev_probs)
            sampled_x_emi_mean_list_features.append(xt_emi_mu_features)
            sampled_x_emi_logvar_list_features.append(xt_emi_logvar_features)
            
        sampled_z_tra_mean = tf.stack(sampled_z_tra_mean_list, axis=1)
        sampled_z_tra_logvar = tf.stack(sampled_z_tra_logvar_list, axis=1)
        sampled_z_inf_mean = tf.stack(sampled_z_inf_mean_list, axis=1)
        sampled_z_inf_logvar = tf.stack(sampled_z_inf_logvar_list, axis=1)
        sampled_x_emi_mean_events = tf.stack(sampled_x_probs_list_events, axis=1)
        sampled_x_emi_mean_features = tf.stack(sampled_x_emi_mean_list_features, axis=1)
        sampled_x_emi_logvar_features = tf.stack(sampled_x_emi_logvar_list_features, axis=1)

        r_tra_params = tf.stack([sampled_z_tra_mean, sampled_z_tra_logvar], axis=-2)
        r_inf_params = tf.stack([sampled_z_inf_mean, sampled_z_inf_logvar], axis=-2)
        r_emi_ev_params = sampled_x_emi_mean_events
        r_emi_ft_params = tf.stack([sampled_x_emi_mean_features, sampled_x_emi_logvar_features], axis=-2)

        return r_tra_params, r_inf_params, r_emi_ev_params, r_emi_ft_params


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
        self.hidden = layers.Dense(ff_dim, name="z_tra_hidden", activation='relu')
        # TODO: Centralize this code
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_tra_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_tra_logvar", activation='softplus')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)
        return z_mean, z_log_var


# https://youtu.be/rz76gYgxySo?t=1483
class InferenceModel(Model):

    def __init__(self, ff_dim):
        super(InferenceModel, self).__init__()
        self.combiner = layers.Concatenate()
        self.hidden = layers.Dense(ff_dim, name="z_inf_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_inf_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_inf_logvar", activation='softplus')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)
        return z_mean, z_log_var


class EmissionFtModel(Model):

    def __init__(self, feature_len):
        super(EmissionFtModel, self).__init__()
        self.hidden = layers.Dense(feature_len, name="x_ft_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(feature_len, name="x_ft_mean", activation=lambda x: 5 * keras.activations.tanh(x))
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="x_ft_logvar", activation='softplus')

    def call(self, inputs):
        z_sample = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var


class EmissionEvModel(Model):

    def __init__(self, feature_len):
        super(EmissionEvModel, self).__init__()
        self.hidden = layers.Dense(feature_len, name="x_ev_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(feature_len, name="x_ev", activation='softmax')

    def call(self, inputs):
        z_sample = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(z_sample)
        return z_mean