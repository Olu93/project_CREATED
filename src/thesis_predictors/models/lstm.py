from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from .model_commons import InputInterface, ModelInterface, DualInput, TokenInput, VectorInput

from thesis_readers.helper.modes import TaskModeType

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class CustomLSTM(ModelInterface):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.embedder = Embedding(self.vocab_len, embed_dim, mask_zero=0)
        self.lstm_layer = LSTM(self.ff_dim, return_sequences=True)
        self.time_distributed_layer = TimeDistributed(Dense(self.vocab_len))
        self.activation_layer = Activation('softmax')

    def construct_feature_vector(self, inputs, embedder):
        features = None
        if type(self.input_interface) is TokenInput:
            indices = inputs
            features = embedder(indices)
        if type(self.input_interface) is DualInput:
            indices, other_features = inputs
            embeddings = embedder(indices)
            features = tf.concat([embeddings, other_features], axis=-1)
        if type(self.input_interface) is VectorInput:
            features = inputs
        return features

    def call(self, inputs):
        x = self.construct_feature_vector(inputs, self.embedder)
        x = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)
        return y_pred


class LSTMModelOnewayNearsight(CustomLSTM):
    task_mode_type = TaskModeType.FIX2ONE
    input_interface = TokenInput()

    def __init__(self, embed_dim=10, ff_dim=10, **kwargs):
        super(LSTMModelOnewayNearsight, self).__init__(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)

    def call(self, inputs):
        x = super().call(inputs)
        x = x[:,-1]
        return x


# https://keras.io/guides/functional_api/
class LSTMModelOneWay(CustomLSTM):
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = TokenInput()

    def __init__(self, embed_dim=10, ff_dim=10, **kwargs):
        super(LSTMModelOneWay, self).__init__(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)


class LSTMModelOneWaySeperated(CustomLSTM):  # TODO: Change to Single and Sequence
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = DualInput()

    def __init__(self, embed_dim=10, ff_dim=10, **kwargs):
        super(LSTMModelOneWaySeperated, self).__init__(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)


class LSTMModelOneWayVector(CustomLSTM):
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = VectorInput()

    def __init__(self, embed_dim=10, ff_dim=10, **kwargs):
        super(LSTMModelOneWayVector, self).__init__(embed_dim=embed_dim, ff_dim=ff_dim, **kwargs)