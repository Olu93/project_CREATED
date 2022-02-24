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




class DMMModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        self.feature_len = embed_dim + self.feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = Sampler(self.ff_dim)
        self.emitter = EmissionModel(self.feature_len)


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        sampled_z_tra_mean_list = []
        sampled_z_tra_logvar_list = []
        sampled_z_inf_mean_list = []
        sampled_z_inf_logvar_list = []
        sampled_x_emi_mean_list = []
        sampled_x_emi_logvar_list = []
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
            xt_emi_mu, xt_emi_logvar = self.emitter(zt_sample)
            
            sampled_z_tra_mean_list.append(z_transition_mu)
            sampled_z_tra_logvar_list.append(z_transition_logvar)
            sampled_z_inf_mean_list.append(z_inf_mu)
            sampled_z_inf_logvar_list.append(z_inf_logvar)
            sampled_x_emi_mean_list.append(xt_emi_mu)
            sampled_x_emi_logvar_list.append(xt_emi_logvar)
            
        sampled_z_tra_mean = tf.stack(sampled_z_tra_mean_list, axis=1)
        sampled_z_tra_logvar = tf.stack(sampled_z_tra_logvar_list, axis=1)
        sampled_z_inf_mean = tf.stack(sampled_z_inf_mean_list, axis=1)
        sampled_z_inf_logvar = tf.stack(sampled_z_inf_logvar_list, axis=1)
        sampled_x_emi_mean = tf.stack(sampled_x_emi_mean_list, axis=1)
        sampled_x_emi_logvar = tf.stack(sampled_x_emi_logvar_list, axis=1)
        return [sampled_x_emi_mean, sampled_x_emi_logvar], [sampled_z_tra_mean, sampled_z_tra_logvar], [sampled_z_inf_mean, sampled_z_inf_logvar]


class DMMnterpretorModel(commons.InterpretorPartMixin):

    def __init__(self, *args, **kwargs):
        super(DMMnterpretorModel, self).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.ff_dim = kwargs.get('ff_dim')
        self.vocab_len = kwargs.get('vocab_len')
        self.lstm_layer = layers.Bidirectional(layers.LSTM(self.ff_dim, return_sequences=True))
        self.output_layer = layers.TimeDistributed(layers.Dense(self.vocab_len))
        self.activation_layer = layers.Softmax()

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = metric.JoinedLoss([metric.MSpCatCE(name="cat_ce")])
        # metrics = []
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        pred_event_probs = inputs
        x = self.lstm_layer(pred_event_probs)
        x = self.output_layer(x)
        x = self.activation_layer(x)
        return x


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
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_tra_mean", activation='relu')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_tra_logvar", activation='relu')

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
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_inf_mean", activation='relu')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_inf_logvar", activation='relu')

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
        # TODO: Maybe remove the 0.5 and include proper log handling
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EmissionModel(Model):

    def __init__(self, feature_len):
        super(EmissionModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(feature_len, name="z_emi_mean", activation='softplus')
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="z_emi_logvar", activation='softplus')

    def call(self, inputs):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var


