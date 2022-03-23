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


class DMMModelCellwise(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModelCellwise, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        # self.feature_len = kwargs["feature_len"]
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.dynamic_vae = layers.RNN(CustomDynamicVAECell(self.ff_dim), return_sequences=True, return_state=True)
        # https://stats.stackexchange.com/a/198047
        self.emitter_ev = ParamBlockLayer(self.vocab_len, axis=2, activation="linear")
        self.emitter_ft = ParamBlockLayer(self.feature_len, axis=2, activation="linear")
        self.sampler = commons.Sampler()
        self.masker = layers.Masking()

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        gt_backwards = self.future_encoder(inputs, training=training, mask=mask)
        results, state = self.dynamic_vae(gt_backwards)
        transition_params = results[:, :, 0]
        inference_params = results[:, :, 1]
        inference_mus, inference_sigmassqs = ParamBlockLayer.split_to_params_seq(inference_params)
        z_emi_samples = self.sampler([inference_mus, inference_sigmassqs])
        x_emission_ev = self.emitter_ev(z_emi_samples)
        x_emission_ft = self.emitter_ft(z_emi_samples)

        return transition_params, inference_params, x_emission_ev, x_emission_ft


# https://stackoverflow.com/questions/54231440/define-custom-lstm-cell-in-keras


class ParamBlockLayer(layers.Layer):

    def __init__(self, units, axis=1, activation='tanh', trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ParamBlockLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.block_mu = keras.Sequential([layers.Dense(units, activation='relu'), layers.Dense(units, activation=activation)])
        self.block_sigmasq = keras.Sequential([layers.Dense(units, activation='relu'), layers.Dense(units, activation=activation)])
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

        self.transition_block = ParamBlockLayer(self.units, axis=1)
        self.inference_block = ParamBlockLayer(self.units, axis=1)

        self.combiner = layers.Concatenate()
        self.sampler = commons.Sampler()

    @property
    def state_size(self):
        return self.units

    def call(self, inputs, states):
        curr_future = inputs
        z_transition_sample = states[0]

        transition_params = self.transition_block(z_transition_sample)
        combined_inference_input = self.combiner([curr_future, z_transition_sample])
        inference_params = self.inference_block(combined_inference_input)
        inference_mus, inference_sigmasqs = ParamBlockLayer.split_to_params_mono(inference_params)
        z_inference_sample = self.sampler([inference_mus, inference_sigmasqs])
        # emission_params_ft = self.emission_block_ft(z_inference_sample)
        # emission_params_ev = self.emission_block_ev(z_inference_sample)
        # results = tf.stack([transition_params, inference_params, emission_params_ft, emission_params_ev], axis=1)
        results = tf.stack([transition_params, inference_params], axis=1)
        return results, z_inference_sample


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
