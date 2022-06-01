import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (LSTM, Activation, Bidirectional, Dense,
                                     Embedding, GlobalAveragePooling1D, Input,
                                     TimeDistributed)
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.base_layer import Layer

from thesis_commons.modes import TaskModeType

from .model_commons import ModelInterface

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# https://blog.paperspace.com/nlp-machine-translation-with-keras/#encoder-architecture-
# https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639

# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/43531172#43531172
# https://keras.io/guides/functional_api/
# https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
class SeqToSeqSimpleLSTMModelOneWay(ModelInterface):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SeqToSeqSimpleLSTMModelOneWay, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.encoder = Encoder(vocab_len, max_len, embed_dim, ff_dim)
        self.decoder = Decoder(vocab_len, max_len, embed_dim, ff_dim)

    def call(self, inputs):
        x_enc, h, c = self.encoder(inputs)
        x_dec = self.decoder([x_enc, h, c])
        return x_dec

    def summary(self):
        x = Input(shape=(self.max_len, ))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Encoder(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(Encoder, self).__init__()
        self.max_len = max_len
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x)
        return output, h, c


class Decoder(Model):
    def __init__(self, vocab_len, max_len, embedding_dim, ff_dim, attention_type='luong'):
        super(Decoder, self).__init__()
        self.attention_type = attention_type
        self.max_len = max_len

        # Define the fundamental cell for decoder recurrent structure
        self.decoder = keras.layers.LSTM(ff_dim, return_sequences=True, return_state=True)

        #Final Dense layer on which softmax will be applied
        self.dense = layers.TimeDistributed(Dense(vocab_len, activation='softmax'))

    def call(self, inputs):
        x_enc, h, c = inputs
        dec_out, _, _ = self.decoder(x_enc, initial_state=[h, c])
        logits = self.dense(dec_out)
        return logits


class SeqToSeqSimpleLSTMModelTwoWay(SeqToSeqSimpleLSTMModelOneWay):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super().__init__(vocab_len, max_len, embed_dim, ff_dim)
        self.encoder = EncoderBi(vocab_len, max_len, embed_dim, ff_dim)

class EncoderBi(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(EncoderBi, self).__init__()
        self.max_len = max_len
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.lstm_layer = layers.Bidirectional(LSTM(ff_dim, return_sequences=True), merge_mode='ave')
        self.lstm_layer_2 = LSTM(ff_dim, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm_layer(x)
        x, h, c = self.lstm_layer_2(x)
        return x, h, c