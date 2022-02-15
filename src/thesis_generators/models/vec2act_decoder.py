from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import CustomEmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin

from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

class Vec2ActDecoder(Model):
    
    def __init__(self, ff_dim, vocab_len, max_len, feature_len, *args, **kwargs):
        super(Vec2ActDecoder).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim, return_states=True))
        self.output_layer = layers.TimeDistributed(layers.Dense(vocab_len))
        self.activation_layer = layers.Softmax()
    
    def call(self, inputs, training=None, mask=None):
        x, h, c = self.lstm_layer(inputs)
        x = self.output_layer(h)
        x = self.activation_layer(x)
        return x
