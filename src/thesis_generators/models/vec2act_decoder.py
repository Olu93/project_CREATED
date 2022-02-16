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
    
    def __init__(self, ff_dim=None, vocab_len=None, max_len=None, feature_len=None, embed_dim=None, *args, **kwargs):
        super(Vec2ActDecoder, self).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim, return_sequences=True))
        self.output_layer = layers.TimeDistributed(layers.Dense(vocab_len))
        self.activation_layer = layers.Softmax()
    
    def call(self, inputs, training=None, mask=None):
        x = self.lstm_layer(inputs) 
        x = self.output_layer(x)
        x = self.activation_layer(x)
        return x
