from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_generators.models.model_commons import MetricTraditional, TokenEmbedder, HybridEmbedder, VectorEmbedder
from thesis_generators.models.model_commons import GeneratorInterface

from thesis_predictors.models.model_commons import HybridInput, VectorInput
import inspect
# https://stackoverflow.com/a/50465583/4162265


class CustomGeneratorVAE(GeneratorInterface):

    def __init__(self, embed_dim, ff_dim, layer_dims=[10, 5, 3], *args, **kwargs):
        print(__class__)
        super(CustomGeneratorVAE, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.encoder_layer_dims = layer_dims
        self.decoder_layer_dims = reversed(layer_dims)
        self.embedder = None
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.sampler = Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(self.vocab_len, self.ff_dim, self.decoder_layer_dims)
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        x = self.embedder(inputs)
        z_mean_and_logvar = self.encoder(x)
        z_sample = self.sampler(z_mean_and_logvar)
        x = self.decoder(z_sample)
        y_pred = self.activation_layer(x)
        return y_pred

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
        inputs = None
        if isinstance(self.embedder, TokenEmbedder):
            inputs = tf.keras.layers.Input(shape=(self.max_len, ))
        if isinstance(self.embedder, HybridEmbedder):
            events = tf.keras.layers.Input(shape=(self.max_len, ))
            features = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
            inputs = [events, features]
        if isinstance(self.embedder, VectorEmbedder):
            inputs = tf.keras.layers.Input(shape=(self.max_len, self.feature_len))
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)


class GeneratorVAETraditional(MetricTraditional, CustomGeneratorVAE, Model):

    def __init__(self, embed_dim, ff_dim, layer_dims=[10, 5, 3], *args, **kwargs):
        print(__class__)
        super(GeneratorVAETraditional, self).__init__(embed_dim=embed_dim, ff_dim=ff_dim, layer_dims=layer_dims, *args, **kwargs)
        # super(MetricTraditional, self).__init__()
        # super(MetricTraditional, self).__init__()
        self.embedder = TokenEmbedder(self.vocab_len, self.embed_dim)


class SeqEncoder(Layer):

    def __init__(self, ff_dim, layer_dims):
        super(SeqEncoder, self).__init__()
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_state=True)
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
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class InnerDecoder(Model):

    def __init__(self, layer_dims):
        super(InnerDecoder, self).__init__()
        self.decode_hidden_state = tf.keras.Sequential([layers.Dense(l_dim) for l_dim in layer_dims])

    def call(self, x):
        x = self.decode_hidden_state(x)
        return x


class SeqDecoder(Layer):

    def __init__(self, vocab_len, ff_dim, layer_dims):
        super(SeqDecoder, self).__init__()
        self.decoder = InnerDecoder(layer_dims)
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len, activation='softmax'))

    def call(self, inputs):
        x = self.decoder(inputs)
        x, h, c = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        return x
