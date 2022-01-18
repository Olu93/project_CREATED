from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine.base_layer import Layer
from keras.utils.vis_utils import plot_model

from thesis_readers.helper.modes import TaskModeType
from .model_commons import ModelInterface
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/43531172#43531172
# https://keras.io/guides/functional_api/
# https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
class SeqToSeqSimpleLSTMModelOneWay(ModelInterface):
    task_mode_type = TaskModeType.MANY2MANY

    
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SeqToSeqSimpleLSTMModelOneWay, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.encoder = Encoder(vocab_len, max_len, embed_dim, ff_dim)
        self.lpad = layers.ZeroPadding1D((1, 0))
        self.rpad = layers.ZeroPadding1D((0, 1))

        self.decoder = Decoder(vocab_len, max_len, embed_dim, ff_dim)

    def call(self, inputs):
        x_enc, h, c = self.encoder(inputs)
        # dec_initial_state = self.decoder.build_initial_state([h, c])
        x_dec = self.decoder(x_enc, [h, c])
        return x_dec

    def summary(self):
        x = Input(shape=(self.max_len, ))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Encoder(tf.keras.Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=  20):
        super(Encoder, self).__init__()
        self.max_len = max_len
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.lstm_layer = tf.keras.layers.LSTM(ff_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x)
        return output, h, c


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, max_len, embedding_dim, ff_dim, attention_type='luong'):
        super(Decoder, self).__init__()
        self.attention_type = attention_type
        self.max_len = max_len

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(ff_dim)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        #Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)
        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_rnn_cell, sampler=self.sampler, output_layer=self.fc)


    def call(self, inputs, initial_state):
        output, state, lengths = self.decoder(inputs, initial_state=initial_state)
        # tf.print("======================================")
        # tf.print(output.shape)
        return output.rnn_output
