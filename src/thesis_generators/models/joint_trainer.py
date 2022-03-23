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


DEBUG_LOSS = False

# https://keras.io/examples/generative/conditional_gan/
# TODO: Implement an LSTM version of this
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
        # self.reverse_embedder_ev = layers.Dense(self.vocab_len, activation='linear')
        # self.reverse_embedder_ft = layers.Dense(self.feature_len, activation='linear')
        print("Instantiate generator...")
        self.generator = GeneratorModel(*args, **kwargs)
        self.custom_loss = SeqProcessLoss()

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

    def train_step(self, data):
        (events_input, features_input), (events_target, features_target) = data
        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            x = self.embedder([events_input, features_input])  # TODO: Dont forget embedding training!!!
            tra_params, inf_params, emi_ev_params, emi_ft_params = self.generator(x)
            # emi_ev_params_splitted, emi_ft_params_splitted = MultiTrainer.split_params(emi_ev_params), MultiTrainer.split_params(emi_ft_params)
            # x_ev_sampled = self.sampler(emi_ev_params_splitted) 
            # x_ev_reversed_emb = self.reverse_embedder_ev(x_ev_sampled)
            # x_ft_sampled = self.sampler(emi_ft_params_splitted) 
            # x_ft_reversed_emb = self.reverse_embedder_ft(x_ft_sampled)
            vars = (tra_params, inf_params, emi_ev_params, emi_ft_params)
            g_loss = self.custom_loss(data[0], vars)
        if tf.math.is_nan(g_loss).numpy():
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(g_loss))}")
        if DEBUG_LOSS:
            total_loss = K.sum([val.numpy() for _, val in self.custom_loss.composites.items()])
            composite_losses = {key:val.numpy() for key, val in self.custom_loss.composites.items()}
            print(f"Total loss is {total_loss} with composition {composite_losses}")

        trainable_weights = self.embedder.trainable_weights + self.generator.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, trainable_weights))
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_loss.composites
        
        return trainer_losses

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        new_x = self.embedder([events, features]) 
        generated_params = self.generator(new_x)
        new_x_rec_events = self.sampler(MultiTrainer.split_params(generated_params[2]))
        new_x_rec_features = self.sampler(MultiTrainer.split_params(generated_params[3]))
        return new_x_rec_events, new_x_rec_features

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:,:,0], input[:,:,1]
        return mus, logsigmas
    
    def get_generator(self) -> Model: 
        return self.generator

    def get_embedder(self) -> Model: 
        return self.embedder

class SeqProcessLoss(metric.JoinedLoss):

    def __init__(self, reduction=keras.losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.NegativeLogLikelihood(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = metric.NegativeLogLikelihood(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.kl_loss = metric.GeneralKLDivergence(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()


    def call(self, y_true, y_pred):
        xt_true_events, xt_true_features = y_true
        xt_true_events_onehot = keras.utils.to_categorical(xt_true_events)
        zt_tra_params, zt_inf_params, zt_emi_ev_params, zt_emi_ft_params= y_pred
        ev_params = SeqProcessLoss.split_params(zt_emi_ev_params)
        ft_params = SeqProcessLoss.split_params(zt_emi_ft_params)
        tra_params = SeqProcessLoss.split_params(zt_tra_params)
        inf_params = SeqProcessLoss.split_params(zt_inf_params)
        rec_loss_events = self.rec_loss_events(xt_true_events_onehot, ev_params)
        rec_loss_features = self.rec_loss_features(xt_true_features, ft_params)
        kl_loss = self.kl_loss(inf_params, tra_params)
        elbo_loss = -(rec_loss_events + rec_loss_features) - kl_loss
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss_events"] = rec_loss_events
        self._losses_decomposed["rec_loss_features"] = rec_loss_features
        self._losses_decomposed["total"] = elbo_loss
        if any([tf.math.is_nan(l).numpy() for k,l in self._losses_decomposed.items()]):
            print(f"Something happened! - There's at least one nan-value")
            rec_loss_events = self.rec_loss_events(xt_true_events_onehot, ev_params)
            rec_loss_features = self.rec_loss_features(xt_true_features, ft_params)
            kl_loss = self.kl_loss(zt_inf_params, zt_tra_params)
            elbo_loss = rec_loss_events + rec_loss_features - kl_loss
        return elbo_loss
    
    
    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:,:,0], input[:,:,1]
        return mus, logsigmas