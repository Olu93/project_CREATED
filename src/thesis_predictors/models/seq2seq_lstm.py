from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine.base_layer import Layer
from keras.utils.vis_utils import plot_model
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/43531172#43531172
# https://keras.io/guides/functional_api/
# https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
class SeqToSeqSimpleLSTMModelOneWay(Model):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SeqToSeqSimpleLSTMModelOneWay, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.encoder = LSTM(ff_dim, return_sequences=True, return_state=True)
        self.concat = layers.Concatenate()
        self.lpad = layers.ZeroPadding1D((1, 0))
        self.rpad = layers.ZeroPadding1D((0, 1))

        self.decoder = LSTM(ff_dim, return_sequences=True, return_state=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        x_enc = self.embedding(inputs[:, :-1])
        x_dec = self.embedding(inputs)
        # x_dec = self.rpad(self.embedding(inputs[:, 1:]))
        h_enc, _, _ = self.encoder(x_enc)

        dec_input = self.concat([h_enc, x_dec])
        h_dec, _, _ = self.decoder(dec_input)
        logits = self.time_distributed_layer(h_dec)
        y_pred = self.activation_layer(logits)
        return y_pred

    def summary(self):
        x = Input(shape=(self.max_len, ))
        model = Model(inputs=[[x]], outputs=self.call([x]))
        return model.summary()


class Encoder(tf.keras.Model):
    def __init__(self, vocab_len=None, embed_dim=10, ff_dim=20):
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
        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(ff_dim)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = tfa.seq2seq.BahdanauAttention(units=ff_dim) if attention_type == 'bahdanau' else tfa.seq2seq.LuongAttention(units=ff_dim)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=ff_dim)

        #Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)
        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_initial_state(self, encoder_state):
        decoder_initial_state = self.rnn_cell.get_initial_state()
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state)
        return outputs
