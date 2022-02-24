from pydoc import classname
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_commons.functions import sample
from thesis_commons import metric
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin

from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, Type, TypeVar, NewType
import thesis_generators.models.model_commons as commons



# https://keras.io/examples/generative/conditional_gan/
class MultiTrainer(Model):
    
    def __init__(self, Embedder: Type[commons.EmbedderLayer], GeneratorModel: Type[commons.GeneratorPartMixin], InterpretorModel: Type[commons.InterpretorPartMixin], *args,
                 **kwargs):
        super(MultiTrainer, self).__init__(name="_".join([cl.__name__ for cl in [type(self), Embedder, GeneratorModel, InterpretorModel]]))
        # Seperately trained
        self.max_len = kwargs.get('max_len')
        self.feature_len = kwargs.get('feature_len')
        self.embed_dim = kwargs.get('embed_dim')
        self.in_events = layers.Input(shape=(self.max_len, ))
        self.in_features = layers.Input(shape=(self.max_len, self.feature_len))
        self.sampler = commons.Sampler()
        print("Instantiate embbedder...")
        self.embedder = Embedder(*args, **kwargs)
        print("Instantiate generator...")
        self.generator = GeneratorModel(*args, **kwargs)
        print("Instantiate interpreter...")
        self.interpreter = InterpretorModel(*args, **kwargs)
        

    def compile(self,
                g_optimizer=None,
                g_loss=None,
                g_metrics=None,
                g_loss_weights=None,
                g_weighted_metrics=None,
                i_optimizer=None,
                i_loss=None,
                i_metrics=None,
                i_loss_weights=None,
                i_weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        self.generator.compile(optimizer=g_optimizer or self.generator.optimizer or tf.keras.optimizers.Adam(),
                               loss=g_loss,
                               metrics=g_metrics,
                               loss_weights=g_loss_weights,
                               weighted_metrics=g_weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
        self.interpreter.compile(optimizer=i_optimizer or self.interpreter.optimizer or tf.keras.optimizers.Adam(),
                                 loss=i_loss,
                                 metrics=i_metrics,
                                 loss_weights=i_loss_weights,
                                 weighted_metrics=i_weighted_metrics,
                                 run_eagerly=run_eagerly,
                                 steps_per_execution=steps_per_execution,
                                 **kwargs)
        default_metrics = [metric.MSpCatAcc(name="cat_acc"), metric.MEditSimilarity(name="ed_sim")]

        return super().compile(optimizer=tf.keras.optimizers.Adam(), metrics=default_metrics, run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, **kwargs)

    # Not needed as all metrics are losses
    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return self.generator.compiled_metrics._user_metrics + self.interpreter.compiled_metrics._user_metrics

    def train_step(self, data):
        (events, features), _ = data
        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            x = self.embedder([events, features])  # TODO: Dont forget embedding training!!!
            generated_vectors = self.generator(x)
            g_loss = self.generator.loss(x, generated_vectors)
        trainable_weights = self.embedder.trainable_weights + self.generator.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, trainable_weights))

        # Train the Decoder.
        with tf.GradientTape() as tape:
            x_sample = sample(generated_vectors[0])
            decoded_sequence_probs = self.interpreter(x_sample)
            i_loss = self.interpreter.loss(events, decoded_sequence_probs)
        grads = tape.gradient(i_loss, self.interpreter.trainable_weights)
        self.interpreter.optimizer.apply_gradients(zip(grads, self.interpreter.trainable_weights))

        new_x = self.embedder([events, features])
        gen_seq_out = self.generator(new_x)
        new_x_rec = self.sampler(gen_seq_out[0])
        int_seq_out = self.interpreter(new_x_rec)
        g_loss = self.generator.loss(x, gen_seq_out)
        i_loss = self.interpreter.loss(events, int_seq_out)
        metrics_collector.update({m: val for m, val in self.generator.loss.composites.items()})
        metrics_collector.update({m: val for m, val in self.interpreter.loss.composites.items()})
        self.compiled_metrics.update_state(events, int_seq_out)
        metrics_collector.update({m.name: m.result() for m in self.metrics})
        return metrics_collector

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        new_x = self.embedder([events, features])
        gen_seq_out = self.generator(new_x)
        new_x_rec = sample(gen_seq_out[0])
        int_seq_out = self.interpreter(new_x_rec)
        return int_seq_out