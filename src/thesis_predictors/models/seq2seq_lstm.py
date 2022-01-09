from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine.base_layer import Layer
from keras.utils.vis_utils import plot_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/43531172#43531172
# https://keras.io/guides/functional_api/
# https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
class SeqToSeqLSTMModelOneWay(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SeqToSeqLSTMModelOneWay, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.encoder = LSTM(ff_dim, return_sequences=True, return_state=True)
        self.concat = layers.Concatenate()
        self.lpad = layers.ZeroPadding1D((1, 0))
        self.rpad = layers.ZeroPadding1D((0, 1))

        self.decoder = LSTM(ff_dim, return_sequences=True, return_state=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        x_enc = self.lpad(self.embedding(inputs[0][:, :-1]))
        x_dec = self.embedding(inputs[0])
        # x_dec = self.rpad(self.embedding(inputs[:, 1:]))
        h_enc, _, _ = self.encoder(x_enc)

        dec_input = self.concat([h_enc, x_dec])
        h_dec, _, _ = self.decoder(dec_input)
        logits = self.time_distributed_layer(h_dec)
        y_pred = self.activation_layer(logits)
        return y_pred

    def summary(self):
        x = Input(shape=(self.max_len,))
        model = Model(inputs=[[x]], outputs=self.call([x]))
        return model.summary()

    def make_predict_function(self):
        return super().make_predict_function()


# class SimpleEncoder(Layer):
#     # https://stackoverflow.com/a/43531172/4162265
#     def __init__(self, vocab_len, max_len, embed_dim, ff_dim):
#         super(SimpleEncoder, self).__init__()

#         self.encoder_layer = LSTM(ff_dim, return_sequences=True, return_state=True)
#         # self.helper = TimeDistributed(Dense(1))
#         # self.encode = layers.AveragePooling1D(max_len)

#     def call(self, inputs):
#         enc = self.embedding(inputs)
#         # enc = self.helper(enc)
#         # tf.print(tf.reduce_sum(tf.cast(tf.not_equal(inputs,0), tf.float32)))
#         # tf.print(inputs.shape)
#         # tf.print(enc.shape)
#         lstm_results = self.encoder_layer(enc)
#         all_state_h, last_state_h, last_state_c = lstm_results
#         return self.encode(all_state_h)

#     def summary(self):
#         x = Input(shape=(self.max_len, self.embed_dim))
#         model = Model(inputs=[x], outputs=self.call(x))
#         return model.summary()

# class Decoder(Layer):
#     # https://stackoverflow.com/a/43531172/4162265
#     def __init__(self, vocab_len, embed_dim, ff_dim):
#         super(Decoder, self).__init__()
#         self.decoder_layer = LSTM(embed_dim, return_sequences=True, return_state=True)
#         self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
#         self.embed_dim = embed_dim

#     def call(self, inputs):
#         tf.print(inputs)
#         lstm_results = self.decoder_layer(inputs)
#         all_state_h, last_state_h, last_state_c = lstm_results
#         return self.time_distributed_layer(all_state_h)

#     def summary(self):
#         x = Input(shape=(self.max_len, self.embed_dim))
#         model = Model(inputs=[x], outputs=self.call(x))
#         return model.summary()


class Seq2SeqLSTMModelBidrectional(SeqToSeqLSTMModelOneWay):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(Seq2SeqLSTMModelBidrectional, self).__init__(vocab_len, max_len, embed_dim=10, ff_dim=20)
        self.encoder_layer = Bidirectional(LSTM(ff_dim, return_state=True))


# class TemporalMeanPooling(Layer):
#     """
#     This is a custom Keras layer. This pooling layer accepts the temporal
#     sequence output by a recurrent layer and performs temporal pooling,
#     looking at only the non-masked portion of the sequence. The pooling
#     layer converts the entire variable-length hidden vector sequence
#     into a single hidden vector, and then feeds its output to the Dense
#     layer.

#     input shape: (nb_samples, nb_timesteps, nb_features)
#     output shape: (nb_samples, nb_features)
#     """
#     def __init__(self, **kwargs):
#         super(TemporalMeanPooling, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.input_spec = [layers.InputSpec(ndim=3)]

#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[2])

#     def call(self, x, mask=None):  #mask: (nb_samples, nb_timesteps)
#         if mask is None:
#             mask = keras.backend.mean(keras.backend.ones_like(x), axis=-1)
#         ssum = keras.backend.sum(x, axis=-2)  #(nb_samples, np_features)
#         mask = keras.backend.cast(mask, keras.backend.floatx())
#         rcnt = keras.backend.sum(mask, axis=-1, keepdims=True)  #(nb_samples)
#         return ssum / rcnt
#         #return rcnt

#     def compute_mask(self, input, mask):
#         return None