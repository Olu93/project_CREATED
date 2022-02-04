from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_generators.models.model_commons import GeneratorInterface

from thesis_predictors.models.model_commons import TokenInput, HybridInput, VectorInput


class CustomLSTM(GeneratorInterface, Model):

    def __init__(self, embed_dim, ff_dim, layer_dims, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.encoder_layer_dims = layer_dims
        self.decoder_layer_dims = reversed(layer_dims)
        self.embedder = SeqEmbedder(self.vocab_len, embed_dim, mask_zero=0)
        self.encoder = SeqEncoder(self.encoder_layer_dims)
        self.sampler = Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(self.decoder_layer_dims)
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.encoder(x)
        x_mean, x_log_var = self.sampler(x)
        x = self.decoder([x_mean, x_log_var])
        y_pred = self.activation_layer(x)
        return y_pred


class SeqEmbedder(Layer):

    def __init__(self, input_interface, **kwargs):
        super(SeqEmbedder, self).__init__(**kwargs)
        self.input_interface = input_interface

    def construct_feature_vector(self, inputs, embedder):
        features = None
        if type(self.input_interface) is TokenInput:
            indices = inputs
            features = embedder(indices)
        if type(self.input_interface) is HybridInput:
            indices, other_features = inputs
            embeddings = embedder(indices)
            features = tf.concat([embeddings, other_features], axis=-1)
        if type(self.input_interface) is VectorInput:
            features = inputs
        return features

    def call(self, inputs, **kwargs):
        x = self.construct_feature_vector(inputs, self.embedder)
        return x


class SeqEncoder(Layer):

    def __init__(self, ff_dim, layer_dims):
        super(SeqEncoder, self).__init__()
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_state=True)
        self.combiner = layers.Concatenate()
        self.encoder = InnerEncoder(layer_dims)
        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        self.latent_log_var = layers.Dense(layer_dims[-1], name="z_log_var")

    def call(self, inputs):
        x, h, c = self.encoder(inputs)
        x = self.combiner([x, h, c])
        return x


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

    def __init__(self, ff_dim, layer_dims):
        super(SeqEncoder, self).__init__()
        self.decoder = InnerDecoder(layer_dims)
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(self.vocab_len, activation='softmax'))

    def call(self, inputs):
        x = self.decoder(inputs)
        x, h, c = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        return x
    
