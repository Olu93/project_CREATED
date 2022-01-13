from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import layers
from thesis_readers.helper.modes import TaskModes

from thesis_predictors.models.model_commons import ModelInterface

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# https://keras.io/guides/functional_api/
class FullLSTMModelOneWay(ModelInterface):
    # name = 'lstm_unidirectional'
    def __init__(self, vocab_len, max_len, feature_len, embed_dim=10, ff_dim=20, *args, **kwargs):
        super(FullLSTMModelOneWay, self).__init__(*args, **kwargs)
        self.max_len = max_len
        self.feature_len = feature_len
        # self.inputs = InputLayer(input_shape=(max_len,))
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.concat = layers.Concatenate()
        self.lstm_layer = LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')
        # self.set_metrics(TaskModes.NEXT_EVENT)

    def call(self, inputs):
        event_ids, features = inputs[0], inputs[1]
        embeddings = self.embedding(event_ids)
        x = self.concat([embeddings, features])
        x = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred

    def summary(self):
        events = Input(shape=(self.max_len,))
        features = Input(shape=(self.max_len, self.feature_len))
        x = [events, features]
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()
