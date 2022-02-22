from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_commons import metrics
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin
import thesis_generators.models.model_commons as commons
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

# def call(self, inputs, training=None, mask=None):
#     x = inputs
#     z_t_minus_1_mean, z_t_minus_1_var = self.state_transitioner(x)
#     z_transition_sampler = self.sampler([z_t_minus_1_mean, z_t_minus_1_var])
#     g_t_backwards = self.future_encoder([z_t_minus_1_mean, z_t_minus_1_var])

#     z_t_sample_minus_1 = self.sampler([g_t_backwards, z_transition_sampler])
#     z_t_mean, z_t_log_var = self.decoder(z_t_sample_minus_1)
#     return z_t_mean, z_t_log_var


class DMMModel(Model):

    def __init__(self, ff_dim, embed_dim, vocab_len, feature_len, max_len, *args, **kwargs):
        print(__class__)
        super(DMMModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        self.vocab_len = vocab_len
        self.feature_len = embed_dim + feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.embedder = commons.HybridEmbedderLayer(vocab_len, embed_dim, *args, **kwargs)
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inference_encoder = InferenceEncoderModel(self.ff_dim)
        self.sampler = Sampler(self.ff_dim)
        self.inference_decoder = InferenceDecoderModel(self.feature_len)
        self.reconstuctor = ReconstructionModel(self.feature_len)
        self.rec_loss = metrics.VAEReconstructionLoss(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.kl_loss = metrics.GeneralKLDivergence(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def train_step(self, data):
        (events, features), _ = data
        metrics_collector = {}
        # Train Process
        x_pred = []
        loss = 0
        kl_part_loss = 0
        rec_part_loss = 0
        with tf.GradientTape() as tape:
            embeddings = self.embedder([events, features])
            gt_backwards = self.future_encoder(embeddings)
            zt_sample = tf.keras.backend.random_uniform(shape=gt_backwards.shape)[:, 0]
            # zt_sample = tf.repeat(self.initial_z, len(gt_backwards), axis=0)
            for t in range(embeddings.shape[1]):
                zt_prev = zt_sample
                gt = gt_backwards[:, t]

                z_transition_mu, z_transition_logvar = self.state_transitioner(zt_prev)
                z_inference_mu, z_inference_logvar = self.inference_encoder([gt, zt_prev])

                zt_sample = self.sampler([z_inference_mu, z_inference_logvar])
                zt_mean, zt_logvar = self.inference_decoder(zt_sample)
                xt_sample = self.reconstuctor([zt_mean, zt_logvar])
                kl_loss = self.kl_loss([z_inference_mu, z_inference_logvar], [z_transition_mu, z_transition_logvar])
                rec_loss = self.rec_loss(embeddings[:, t], xt_sample)
                loss += kl_loss + rec_loss
                kl_part_loss += kl_loss
                rec_part_loss += rec_loss

        all_models = [self.embedder, self.future_encoder, self.state_transitioner, self.inference_encoder, self.inference_decoder, self.reconstuctor]
        trainable_weights = [w for md in all_models for w in md.trainable_weights]
        grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))
        metrics_collector["loss"] = loss
        metrics_collector["kl_loss"] = kl_part_loss
        metrics_collector["rec_loss"] = rec_part_loss
        return metrics_collector

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        embeddings = self.embedder([events, features])
        gt_backwards = self.future_encoder(embeddings)

        zt_sample = tf.keras.backend.random_uniform(shape=gt_backwards.shape)[:, 0]
        # zt_sample = tf.repeat(self.initial_z, len(gt_backwards), axis=0)
        for t in range(embeddings.shape[1]):
            zt_prev = zt_sample
            gt = gt_backwards[:, t]

            z_transition_mu, z_transition_logvar = self.state_transitioner(zt_prev)
            z_inference_mu, z_inference_logvar = self.inference_encoder([gt, zt_prev])

            zt_sample = self.sampler([z_inference_mu, z_inference_logvar])
            zt_mean, zt_logvar = self.inference_decoder(zt_sample)
            xt_sample = self.reconstuctor([zt_mean, zt_logvar])
            kl_loss = self.kl_loss([z_inference_mu, z_inference_logvar], [z_transition_mu, z_transition_logvar])
            rec_loss = self.rec_loss(embeddings[:, t], xt_sample)
            # print(kl_loss.numpy(),rec_loss.numpy())
        return xt_sample


# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(Model):

    def __init__(self, ff_dim):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, go_backwards=True)
        self.combiner = layers.Concatenate()

    def call(self, inputs):
        x = inputs
        x = self.lstm_layer(x)
        # g_t_backwards = self.combiner([h, c])
        return x


# https://youtu.be/rz76gYgxySo?t=1450
class TransitionModel(Model):

    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        z_t_minus_1 = inputs
        z_mean = self.latent_vector_z_mean(z_t_minus_1)
        z_log_var = self.latent_vector_z_log_var(z_t_minus_1)
        return z_mean, z_log_var


# https://youtu.be/rz76gYgxySo?t=1483
class InferenceEncoderModel(Model):

    def __init__(self, ff_dim):
        super(InferenceEncoderModel, self).__init__()
        self.combiner = layers.Concatenate()
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_log_var")

    def call(self, inputs, training=None, mask=None):
        g_t_backwards, z_t_minus_1 = inputs
        combined_input = self.combiner([g_t_backwards, z_t_minus_1])
        z_mean = self.latent_vector_z_mean(combined_input)
        z_log_var = self.latent_vector_z_log_var(combined_input)
        return z_mean, z_log_var


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # TODO: Maybe remove the 0.5 and include proper log handling
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class InferenceDecoderModel(Model):

    def __init__(self, feature_len):
        super(InferenceDecoderModel, self).__init__()
        self.latent_vector_z_mean = layers.Dense(feature_len, name="z_mean")
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="z_log_var")

    def call(self, inputs):
        z_sample = inputs

        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var


class ReconstructionModel(Model):

    def __init__(self, feature_len):
        super(ReconstructionModel, self).__init__()
        self.sampler = Sampler(feature_len)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        x_reconstructed = self.sampler([z_mean, z_log_var])
        return x_reconstructed
