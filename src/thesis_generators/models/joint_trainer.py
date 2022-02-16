from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_commons import metrics
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import CustomEmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin

from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, Type, TypeVar, NewType


class JointTrainer(GeneratorModelMixin, Model):

    def __init__(self, Embedder: Type[Model], GeneratorModel: Type[Model], VecToActModel: Type[Model], *args, **kwargs):
        super(JointTrainer, self).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.max_len = kwargs.get('max_len')
        self.feature_len = kwargs.get('feature_len')
        self.embed_dim = kwargs.get('embed_dim')
        self.in_events = layers.Input(shape=(self.max_len, ))
        self.in_features = layers.Input(shape=(self.max_len, self.feature_len))
        self.embedder = Embedder(*args, **kwargs)
        self.generator = GeneratorModel(*args, **kwargs)
        self.vec2act = VecToActModel(*args, **kwargs)
        self.loss_ce = metrics.MSpCatCE()
        self.metric_acc = metrics.MSpCatAcc()

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        x = self.embedder([events, features])
        generator_output = self.generator(x, training=training)
        pred_event_probs = self.vec2act(generator_output, training=training)
        ce_loss = self.loss_ce(events, pred_event_probs)
        self.add_loss(ce_loss)
        self.add_metric(ce_loss, "cat_ce")
        acc_metric = self.metric_acc(events, pred_event_probs)
        pred_act = keras.backend.argmax(pred_event_probs)
        self.add_metric(acc_metric, "cat_acc")
        return generator_output, pred_act

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)