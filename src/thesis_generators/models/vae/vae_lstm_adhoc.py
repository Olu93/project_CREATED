from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import CustomEmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin

from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265


class CustomGeneratorVAE(GeneratorModelMixin):

    def __init__(self, ff_dim, layer_dims=[13, 8, 5], *args, **kwargs):
        print(__class__)
        super(CustomGeneratorVAE, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        layer_dims = [kwargs.get("feature_len") + kwargs.get("embed_dim")] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.decoder_layer_dims = reversed(layer_dims)
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.sampler = Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(layer_dims[0], self.max_len, self.ff_dim, self.decoder_layer_dims)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        z_mean, z_log_var = self.encoder(x)
        z_sample = self.sampler([z_mean, z_log_var])
        z = self.decoder(z_sample)
        losses = self.compute_loss(inputs, z, z_mean, z_log_var)
        cum_loss = 0
        for name, loss in losses.items():
            cum_loss += loss
            self.add_metric(loss, name=name)
        self.add_loss(cum_loss)
        # if training is not None:
        #     losses = self.compute_loss(inputs, y_pred, z_mean, z_log_var)
        #     cum_loss = 0
        #     for name, loss in losses.items():
        #         cum_loss += loss
        #         self.add_metric(loss, name=name)
        #     self.add_loss(cum_loss)
        return z

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = loss or self.loss
        metrics = metrics or self.metrics
        optimizer = optimizer or self.optimizer or Adam()
        return super().compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = self.in_layer.in_layer_shape
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)


class GeneratorVAETraditional(MetricVAEMixin, CustomGeneratorVAE, Model):

    def __init__(self, *args, **kwargs):
        print(__class__)
        super(GeneratorVAETraditional, self).__init__(*args, **kwargs)


class SeqEncoder(Model):

    def __init__(self, ff_dim, layer_dims):
        super(SeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_state=True)
        self.combiner = layers.Concatenate()
        self.encoder = InnerEncoder(layer_dims)
        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        self.latent_log_var = layers.Dense(layer_dims[-1], name="z_log_var")

    def call(self, inputs):
        x, h, c = self.lstm_layer(inputs)
        x = self.combiner([x, h, c])
        x = self.encoder(x)
        z_mean = self.latent_mean(x)
        z_log_var = self.latent_log_var(x)
        return z_mean, z_log_var


class InnerEncoder(Layer):

    def __init__(self, layer_dims):
        super(InnerEncoder, self).__init__()
        self.encode_hidden_state = tf.keras.Sequential([layers.Dense(l_dim) for l_dim in layer_dims])

    def call(self, inputs):
        x = inputs
        x = self.encode_hidden_state(x)
        return x


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # TODO: Maybe remove the 0.5 and include proper log handling
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class InnerDecoder(layers.Layer):

    def __init__(self, layer_dims):
        super(InnerDecoder, self).__init__()
        self.decode_hidden_state = tf.keras.Sequential([layers.Dense(l_dim) for l_dim in layer_dims])

    def call(self, x):
        # tf.print(x.shape)
        x = self.decode_hidden_state(x)
        return x


class SeqDecoder(Model):

    def __init__(self, in_dim, max_len, ff_dim, layer_dims):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.decoder = InnerDecoder(layer_dims)
        self.repeater = layers.RepeatVector(max_len)
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True)
        self.output_layer = layers.TimeDistributed(layers.Dense(in_dim))

    def call(self, inputs):
        z_sample = inputs
        z_state = self.decoder(z_sample)
        z_input = self.repeater(z_state)
        # x = tf.expand_dims(x,1)
        # z_expanded = tf.repeat(tf.expand_dims(z, 1), self.max_len, axis=1)
        x = self.lstm_layer(z_input)
        x = self.output_layer(x)
        return x
