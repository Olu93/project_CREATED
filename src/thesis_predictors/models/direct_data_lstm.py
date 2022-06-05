import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (LSTM, Activation, Dense, Embedding, Input,
                                     TimeDistributed)
from tensorflow.python.keras import layers

from thesis_commons.modes import TaskModeType
from thesis_predictors.models.model_commons import ModelInterface

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# https://keras.io/guides/functional_api/
class FullLSTMModelOneWayExtensive(ModelInterface):
    # name = 'lstm_unidirectional'
    task_mode_type = TaskModeType.FIX2FIX
    
    def __init__(self, vocab_len, max_len, feature_len, embed_dim=10, ff_dim=20, *args, **kwargs):
        super(FullLSTMModelOneWayExtensive, self).__init__(*args, **kwargs)
        self.max_len = max_len
        self.feature_len = feature_len
        # self.inputs = InputLayer(input_shape=(max_len,))
        self.embedding = Embedding(vocab_len, embed_dim, mask_zero=0)
        self.concat = layers.Concatenate()
        self.lstm_layer = LSTM(ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(vocab_len))
        self.activation_layer = Activation('softmax')

    def call(self, inputs):
        event_ids, features = inputs[0], inputs[1]
        embeddings = self.embedding(event_ids)
        x = self.concat([embeddings, features])
        x = self.lstm_layer(x)
        if self.time_distributed_layer:
            x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred

    def summary(self):
        events = Input(shape=(self.max_len,))
        features = Input(shape=(self.max_len, self.feature_len))
        x = [events, features]
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class FullLSTMModelOneWaySimple(FullLSTMModelOneWayExtensive): # TODO: Change to Single and Sequence
    task_mode_type = TaskModeType.FIX2ONE
    
    def __init__(self, vocab_len, max_len, feature_len, embed_dim=10, ff_dim=20, *args, **kwargs):
        super().__init__(vocab_len, max_len, feature_len, embed_dim=embed_dim, ff_dim=ff_dim, *args, **kwargs)
        self.lstm_layer = LSTM(vocab_len, return_sequences=False)
        # self.time_distributed_layer = Dense(vocab_len)
        self.time_distributed_layer = None