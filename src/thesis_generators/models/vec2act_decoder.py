from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_commons import metrics
# TODO: Fix imports by collecting all commons
import thesis_generators.models.model_commons as commons
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType
import tensorflow.keras.backend as K


class Vec2ActDecoder(Model):

    def __init__(self, ff_dim=None, vocab_len=None, max_len=None, feature_len=None, embed_dim=None, *args, **kwargs):
        super(Vec2ActDecoder, self).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim, return_sequences=True))
        self.output_layer = layers.TimeDistributed(layers.Dense(vocab_len))
        self.activation_layer = layers.Softmax()
        self.loss_ce = metrics.MSpCatCE()

    def call(self, inputs, training=None, mask=None):
        x = self.lstm_layer(inputs)
        x = self.output_layer(x)
        x = self.activation_layer(x)
        return x


class SimpleInterpretorModel(commons.InterpretorPartMixin):

    def __init__(self, *args, **kwargs):
        super(SimpleInterpretorModel, self).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.ff_dim = kwargs.get('ff_dim')
        self.vocab_len = kwargs.get('vocab_len')
        self.lstm_layer = layers.Bidirectional(layers.LSTM(self.ff_dim, return_sequences=True))
        self.output_layer = layers.TimeDistributed(layers.Dense(self.vocab_len))
        self.activation_layer = layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        pred_event_probs = inputs
        x = self.lstm_layer(pred_event_probs)
        x = self.output_layer(x)
        x = self.activation_layer(x)
        return x
