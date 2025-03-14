
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers

import thesis_commons.model_commons as commons
# TODO: Fix imports by collecting all commons
from thesis_commons.model_commons import CustomInputLayer


class VRNNModel(commons.TensorflowModelMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(VRNNModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        # self.feature_len = kwargs["feature_len"]
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.dynamic_vae = layers.RNN(CustomDynamicVAECell(self.ff_dim), return_sequences=True, return_state=True)
        self.combiner = layers.Concatenate()
        self.repeater = layers.RepeatVector(3)

        # https://stats.stackexchange.com/a/198047
        self.emitter_ev = CategoricalBlockLayer(self.vocab_len, axis=2)
        self.emitter_ft = SeqGaussianParamLayer(self.feature_len, axis=2, activation=lambda x: 5 * keras.activations.tanh(x))
        self.sampler = commons.Sampler()
        self.masker = layers.Masking()

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        gt_backwards = self.future_encoder(inputs, training=training, mask=mask)
        results, _, _, _ = self.dynamic_vae(gt_backwards)
        transition_params = results[:, :, 0]
        inference_params = results[:, :, 1]
        emitter_params = K.prod(results[:, :, 2], axis=-2)
        x_emission_ev = self.emitter_ev(emitter_params)
        x_emission_ft = self.emitter_ft(emitter_params)

        return transition_params, inference_params, x_emission_ev, x_emission_ft


# https://stackoverflow.com/questions/54231440/define-custom-lstm-cell-in-keras


class CategoricalBlockLayer(layers.Layer):

    def __init__(self, units, axis=1, activation='softmax', trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(CategoricalBlockLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.feedforward = keras.Sequential([layers.LSTM(units, activation='relu', return_sequences=True), layers.LSTM(units, activation=activation, return_sequences=True)])
        self.axis = axis

    def call(self, inputs, **kwargs):
        probabilities = self.feedforward(inputs, **kwargs)
        return probabilities


class SeqGaussianParamLayer(layers.Layer):

    def __init__(self, units, axis=1, activation='linear', trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(SeqGaussianParamLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.block_hidden = layers.LSTM(units, activation='relu', return_sequences=True)
        self.block_mu = keras.Sequential([self.block_hidden, layers.Dense(units, activation=activation)])
        self.block_sigmasq = keras.Sequential([self.block_hidden, layers.Dense(units, activation='softplus')])
        self.axis = axis

    def call(self, inputs, **kwargs):
        mu = self.block_mu(inputs, **kwargs)
        sigmasq = self.block_sigmasq(inputs, **kwargs)
        return tf.stack([mu, sigmasq], axis=self.axis)


class GaussianParamLayer(layers.Layer):

    def __init__(self, units, axis=1, activation='linear', trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(GaussianParamLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        block_hidden = layers.Dense(units, activation='relu')
        self.block_mu = keras.Sequential([block_hidden, layers.Dense(units, activation=activation)])
        self.block_sigmasq = keras.Sequential([block_hidden, layers.Dense(units, activation='softplus')])
        self.axis = axis

    def call(self, inputs, **kwargs):
        mu = self.block_mu(inputs, **kwargs)
        sigmasq = self.block_sigmasq(inputs, **kwargs)
        return tf.stack([mu, sigmasq], axis=self.axis)

    @staticmethod
    def split_to_params_seq(params):
        mus, logsigmasqs = params[:, :, 0], params[:, :, 1]
        return (mus, logsigmasqs)

    @staticmethod
    def split_to_params_mono(params):
        mus, logsigmasqs = params[:, 0], params[:, 1]
        return (mus, logsigmasqs)


class CustomDynamicVAECell(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units
        super(CustomDynamicVAECell, self).__init__(**kwargs)

        self.transition_block = GaussianParamLayer(self.units, axis=1)
        self.inference_block = GaussianParamLayer(self.units, axis=1)
        # self.state_computer = layers.Dense(self.units)
        self.state_computer = layers.LSTMCell(self.units)
        # self.repeater = layers.RepeatVector(2)
        self.combiner = layers.Concatenate()
        self.sampler = commons.Sampler()

    @property
    def state_size(self):
        return self.units, self.units, self.units

    def call(self, inputs, states):
        x = inputs
        z, h, c = states

        transition_params = self.transition_block(h)

        combined_in = self.combiner([x, h])
        inference_params = self.inference_block(combined_in)

        inference_mus, inference_sigmasqs = GaussianParamLayer.split_to_params_mono(inference_params)

        z_next = self.sampler([inference_mus, inference_sigmasqs])
        combined_in = self.combiner([x, z_next])
        out, (h_next, c_next) = self.state_computer(combined_in, (h, c))

        # results = tf.stack([transition_params, inference_params, CustomDynamicVAECell.dublicate_vector(z_next), CustomDynamicVAECell.dublicate_vector(h_next)], axis=1)
        emitter_params = tf.stack([z_next, h_next], axis=1)
        results = tf.stack([transition_params, inference_params, emitter_params], axis=1)
        return results, (z_next, h_next, c_next)

    # @staticmethod
    # def dublicate_vector(vec, axis=1):
    #     result = tf.stack([vec, vec], axis=axis)
    #     return result


# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(Model):

    def __init__(self, ff_dim):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, go_backwards=True)
        self.combiner = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.lstm_layer(x)
        # g_t_backwards = self.combiner([h, c])
        return x
