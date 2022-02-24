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
from thesis_commons import metric
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265



class SimpleSeqVAEGeneratorModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, layer_dims=[13, 8, 5], *args, **kwargs):
        print(__class__)
        super(SimpleSeqVAEGeneratorModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        layer_dims = [kwargs.get("feature_len") + kwargs.get("embed_dim")] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.decoder_layer_dims = reversed(layer_dims)
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.sampler = Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(layer_dims[0], self.max_len, self.ff_dim, self.decoder_layer_dims)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        z_mean, z_logvar = self.encoder(x)
        z_sample = self.sampler([z_mean, z_logvar])
        x_rec = self.decoder(z_sample)
        return x_rec, z_mean, z_logvar


class SimpleInterpretorModel(commons.InterpretorPartMixin):

    def __init__(self, *args, **kwargs):
        super(SimpleInterpretorModel, self).__init__(*args, **kwargs)
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
