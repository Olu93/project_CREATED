
import tensorflow as tf
from tensorflow.keras import Model, layers

import thesis_commons.model_commons as commons
# TODO: Fix imports by collecting all commons
from thesis_commons.model_commons import CustomInputLayer


class DMMModelSequencewise(commons.TensorflowModelMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModelSequencewise, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
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

    # def build(self, input_shape):
    #     self.zt_init_sample = K.zeros(input_shape)
    #     # return super().build(input_shape)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):

        # xt = self.future_encoder(inputs, training, mask)
        # zt_sample = self.sampler([tf.zeros_like(xt), tf.zeros_like(xt)])
        # z_transition_mu, z_transition_logvar = self.state_transitioner(zt_sample, training, mask)
        # z_inf_mu, z_inf_logvar = self.inferencer([xt, zt_sample], training, mask)
        # zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
        # xt_emi_mu_events = self.emitter_events(zt_sample, training, mask)
        # xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample, training, mask)

        # r_tra_params = tf.stack([z_transition_mu, z_transition_logvar], axis=-2)
        # r_inf_params = tf.stack([z_inf_mu, z_inf_logvar], axis=-2)
        # r_emi_ev_params = xt_emi_mu_events
        # r_emi_ft_params = tf.stack([xt_emi_mu_features, xt_emi_logvar_features], axis=-2)
        # return r_tra_params, r_inf_params, r_emi_ev_params, r_emi_ft_params
        xt = self.future_encoder(inputs, training, mask)
        zt_sample = self.sampler([tf.zeros_like(xt), tf.zeros_like(xt)])
        z_transition_mu, z_transition_logvar = self.state_transitioner(zt_sample, training, mask)
        z_inf_mu, z_inf_logvar = self.inferencer([xt, zt_sample], training, mask)
        zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
        xt_emi_mu_events = self.emitter_events(zt_sample, training, mask)
        xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample, training, mask)

        r_tra_params = tf.stack([z_transition_mu, z_transition_logvar], axis=-2)
        r_inf_params = tf.stack([z_inf_mu, z_inf_logvar], axis=-2)
        r_emi_ev_params = xt_emi_mu_events
        r_emi_ft_params = tf.stack([xt_emi_mu_features, xt_emi_logvar_features], axis=-2)
        return r_tra_params, r_inf_params, r_emi_ev_params, r_emi_ft_params



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
        z_logvar = self.latent_vector_z_log_var(z_t_minus_1, training=training, mask=mask)
        return z_mean, z_logvar


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
        z_logvar = self.latent_vector_z_log_var(combined_input, training=training, mask=mask)
        return z_mean, z_logvar





class EmissionFtModel(Model):

    def __init__(self, feature_len):
        super(EmissionFtModel, self).__init__()
        self.latent_vector_z_mean = layers.LSTM(feature_len, name="z_emi_mean", activation='tanh', return_sequences=True)
        self.latent_vector_z_log_var = layers.LSTM(feature_len, name="z_emi_logvar", activation='tanh', return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample, training=training, mask=mask)
        z_log_var = self.latent_vector_z_log_var(z_sample, training=training, mask=mask)
        return z_mean, z_log_var


class EmissionEvModel(Model):

    def __init__(self, feature_len):
        super(EmissionEvModel, self).__init__()
        self.hidden = layers.LSTM(feature_len, name="x_ev_hidden", activation='relu', return_sequences=True)
        self.latent_vector_z_mean = layers.LSTM(feature_len, name="x_ev", activation='softmax', return_sequences=True)

    def call(self, inputs, training=None, mask=None):
        z_sample = self.hidden(inputs,training=training, mask=mask)
        z_mean = self.latent_vector_z_mean(z_sample, training=training, mask=mask)
        return z_mean