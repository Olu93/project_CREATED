from pydoc import classname
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
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

    def __init__(self, Embedder: Type[commons.EmbedderLayer], GeneratorModel: Type[commons.GeneratorPartMixin], *args, **kwargs):
        super(MultiTrainer, self).__init__(name="_".join([cl.__name__ for cl in [type(self), Embedder, GeneratorModel]]))
        # Seperately trained
        self.max_len = kwargs.get('max_len')
        self.feature_len = kwargs.get('feature_len')
        self.embed_dim = kwargs.get('embed_dim')
        self.vocab_len = kwargs.get('vocab_len')
        self.in_events = layers.Input(shape=(self.max_len, ))
        self.in_features = layers.Input(shape=(self.max_len, self.feature_len))
        self.sampler = commons.Sampler()
        print("Instantiate embbedder...")
        self.embedder = Embedder(*args, **kwargs)
        # self.reverse_embedder = commons.ReverseEmbedding(embedding_layer=self.embedder, *args, **kwargs)
        self.reverse_embedder = layers.Dense(self.vocab_len, activation='softmax')
        print("Instantiate generator...")
        self.generator = GeneratorModel(*args, **kwargs)
        self.custom_loss = SeqProcessLoss()

    # def compute_loss(self, y_true, y_pred):
    #     x_events, x_feat = y_true
    #     x_pred_events_params, x_pred_features_params, z_tra_params, z_emi_params = y_pred
    #     ev_loss = self.ev_loss(x_events, x_pred_events_params)
    #     ft_loss = self.ev_loss(x_feat, x_pred_features_params)
    #     kl_loss = self.kl_loss(z_emi_params, z_tra_params)
    #     return ev_loss, ft_loss, kl_loss

    def compile(self, g_optimizer=None, g_loss=None, g_metrics=None, g_loss_weights=None, g_weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        self.generator.compile(optimizer=g_optimizer or self.generator.optimizer or tf.keras.optimizers.Adam(),
                               loss=g_loss,
                               metrics=g_metrics,
                               loss_weights=g_loss_weights,
                               weighted_metrics=g_weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
        # default_metrics = [metric.MSpCatAcc(name="cat_acc"), metric.MEditSimilarity(name="ed_sim")]

        return super().compile(optimizer=tf.keras.optimizers.Adam(),  run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, **kwargs)

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
        (events_input, features_input), (events_target, features_target) = data
        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            x = self.embedder([events_input, features_input])  # TODO: Dont forget embedding training!!!
            generated_params = self.generator(x)
            x_looked_up = self.reverse_embedder(sample(generated_params[0]))
            vars = (x_looked_up,) + generated_params[1:] 
            g_loss = self.custom_loss(data[0], vars)
        if True:
            tmp_1 = tf.reduce_sum([val.numpy() for _, val in self.custom_loss.composites.items()])
            tmp_2 = [val.numpy() for _, val in self.custom_loss.composites.items()]
            tmp_3 = {key:val.numpy() for key, val in self.custom_loss.composites.items()}

        trainable_weights = self.embedder.trainable_weights + self.reverse_embedder.trainable_weights + self.generator.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, trainable_weights))

        # g_loss = self.custom_loss(data[0], generated_vectors)
        # metrics_collector.update({
        #     "kl_loss": kl_loss,
        #     "ev_loss": ev_loss,
        #     "ft_loss": ft_loss,
        # })

        return self.custom_loss.composites

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        new_x = self.embedder([events, features])
        gen_seq_out = self.generator(new_x)
        new_x_rec_events = sample(gen_seq_out[0])
        new_x_rec_features = sample(gen_seq_out[1])
        return new_x_rec_events, new_x_rec_features
    
class SeqProcessLoss(metric.JoinedLoss):

    def __init__(self, reduction=keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = keras.losses.BinaryCrossentropy(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = metric.NegativeLogLikelihood(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.kl_loss = metric.GeneralKLDivergence(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()


    def call(self, y_true, y_pred):
        xt_true_events, xt_true_features = y_true
        xt_true_events_onehot = keras.utils.to_categorical(xt_true_events)
        xt_events_params, xt_features_params, zt_transition_params, zt_inference_params = y_pred
        sampled_xt_features = self.sampler(xt_features_params)
        rec_loss_events = self.rec_loss_events(xt_true_events_onehot, xt_events_params)
        rec_loss_features = self.rec_loss_features(xt_true_features, xt_features_params)
        kl_loss = self.kl_loss(zt_inference_params, zt_transition_params)
        elbo_loss = rec_loss_events + rec_loss_features - kl_loss
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss_events"] = rec_loss_events
        self._losses_decomposed["rec_loss_features"] = rec_loss_features
        self._losses_decomposed["total"] = elbo_loss
        if any([tf.math.is_nan(l).numpy() for k,l in self._losses_decomposed.items()]):
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(xt_events_params))}")
        return elbo_loss