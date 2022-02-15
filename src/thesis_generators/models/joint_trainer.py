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

class JointTrainer(Model):
    
    def __init__(self, GeneratorModel:Model, VecToActModel:Model, *args, **kwargs):
        super(JointTrainer).__init__(*args, **kwargs)
        # Either trainined in conjunction to generator or seperately
        self.generator = GeneratorModel(*args, **kwargs)
        self.vec2act = VecToActModel(*args, **kwargs)
    
    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        generator_output = self.generator([events, features])
        decoded_activities = self.vec2act(generator_output)
        return decoded_activities
