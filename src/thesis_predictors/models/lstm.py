from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from thesis_predictors.models.model_commons import ModelInterface

from thesis_readers.helper.modes import TaskModeType

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# https://keras.io/guides/functional_api/
class SimpleLSTMModelOneWayExtensive(ModelInterface):
    # name = 'lstm_unidirectional'
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SimpleLSTMModelOneWayExtensive, self).__init__()
        self.max_len = max_len
        # self.masking = layers.Masking(input_shape=(max_len,))
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        # self.lstm_layer = Bidirectional(LSTM(ff_dim, return_sequences=True))
        self.lstm_layer = LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        # x = self.inputs(inputs)
        inputs = inputs
        # x = self.masking(inputs)
        x = self.embedding(inputs)
        x = self.lstm_layer(x)
        if self.time_distributed_layer:
            x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred

    def summary(self):
        x = Input(shape=(self.max_len, ))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class SimpleLSTMModelOneWaySimple(SimpleLSTMModelOneWayExtensive):  # TODO: Change to Single and Sequence
    task_mode_type = TaskModeType.FIX2ONE

    def __init__(self, ff_dim=20, *args, **kwargs):
        super(SimpleLSTMModelOneWaySimple, self).__init__(*args, **kwargs)
        self.lstm_layer = LSTM(ff_dim, return_sequences=False)
        self.time_distributed_layer = None


class SimpleLSTMModelTwoWay(SimpleLSTMModelOneWayExtensive):
    # name = 'lstm_bidirectional'
    # Makes no sense
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=20):
        super(SimpleLSTMModelTwoWay, self).__init__(vocab_len, max_len, embed_dim=10, ff_dim=20)
        self.lstm_layer = Bidirectional(LSTM(ff_dim, return_sequences=True))