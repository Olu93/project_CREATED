import pathlib
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils
import tensorflow as tf
from thesis_generators.models.model_commons import HybridEmbedderLayer
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import BaseModelMixin
import thesis_generators.models.model_commons as commons
from thesis_commons import metric
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class SimpleGeneratorModel(commons.TensorflowModelMixin):
    def __init__(self, ff_dim, layer_dims=[13, 8, 5], *args, **kwargs):
        print(__class__)
        super(SimpleGeneratorModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        layer_dims = [kwargs.get("feature_len") + kwargs.get("embed_dim")] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.embedder = HybridEmbedderLayer(*args, **kwargs)
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.sampler = commons.Sampler()
        self.decoder = SeqDecoder(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        self.custom_loss = SeqProcessLoss(losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.custom_eval = SeqProcessEvaluator()

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        events_input, features_input = inputs
        x = self.embedder([events_input, features_input])
        z_mean, z_logvar = self.encoder(x)
        z_sample = self.sampler([z_mean, z_logvar])
        x_evs, x_fts = self.decoder(z_sample)
        return x_evs, x_fts

    def train_step(self, data):
        if len(data) == 3:
            (events_input, features_input), (events_target, features_target), sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), (events_target, features_target) = data

        with tf.GradientTape() as tape:
            x = self.embedder([events_input, features_input])
            z_mean, z_logvar = self.encoder(x)
            z_sample = self.sampler([z_mean, z_logvar])
            x_evs, x_fts = self.decoder(z_sample)
            vars = [x_evs, x_fts, z_sample, z_mean, z_logvar]  # rec_ev, rec_ft, z_sample, z_mean, z_logvar
            g_loss = self.custom_loss(y_true=[events_target, features_target], y_pred=vars, sample_weight=sample_weight)

        # if tf.math.is_nan(g_loss).numpy():
        #     print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(g_loss))}")
        # if DEBUG_LOSS:
        #     composite_losses = {key: val.numpy() for key, val in self.custom_loss.composites.items()}
        #     print(f"Total loss is {composite_losses.get('total')} with composition {composite_losses}")

        trainable_weights = self.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        eval_loss = self.custom_eval(data[1], vars)
        if tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy():
            print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_eval.composites
        losses = {}
        if DEBUG_SHOW_ALL_METRICS:
            losses.update(trainer_losses)
        losses.update(sanity_losses)
        return losses

    def test_step(self, data):
        # Unpack the data
        if len(data) == 3:
            (events_input, features_input), (events_target, features_target), sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), (events_target, features_target) = data  # Compute predictions
        x = self.embedder([events_input, features_input])
        z_mean, z_logvar = self.encoder(x)
        z_sample = self.sampler([z_mean, z_logvar])
        x_evs, x_fts = self.decoder(z_sample)
        vars = [x_evs, x_fts, z_sample, z_mean, z_logvar]  # rec_ev, rec_ft, z_sample, z_mean, z_logvar        # Updates the metrics tracking the loss
        eval_loss = self.custom_eval(data[1], vars)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        losses = {}
        sanity_losses = self.custom_eval.composites
        sanity_losses["loss"] = 1 - sanity_losses["edit_distance"]  + sanity_losses["feat_mape"] 
        losses.update(sanity_losses)
        return losses

    def sample(self, events_input, features_input, num=10):
        collected_evs, collected_fts = [], []
        for i in range(num):
            x = self.embedder([events_input, features_input])
            z_mean, z_logvar = self.encoder(x)
            z_sample = self.sampler([z_mean, z_logvar])
            x_evs, x_fts = self.decoder(z_sample)
            collected_evs.append(x_evs)
            collected_fts.append(x_fts)
        cf_evs = tf.stack(collected_evs)
        cf_fts = tf.stack(collected_evs)

        return cf_evs, cf_fts

    @staticmethod
    def get_loss_and_metrics():
        return [SeqProcessLoss(losses.Reduction.SUM_OVER_BATCH_SIZE), SeqProcessEvaluator()]


class SeqEncoder(models.Model):
    def __init__(self, ff_dim, layer_dims, max_len):
        super(SeqEncoder, self).__init__()
        # self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, return_state=True)
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim))
        # self.combiner = layers.Concatenate()
        # self.repeater = layers.RepeatVector(max_len)
        self.encoder = InnerEncoder(layer_dims)
        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        self.latent_log_var = layers.Dense(layer_dims[-1], name="z_logvar")

    def call(self, inputs):
        x = self.lstm_layer(inputs)

        x = self.encoder(x)
        z_mean = self.latent_mean(x)
        z_logvar = self.latent_log_var(x)
        return z_mean, z_logvar


class InnerEncoder(layers.Layer):
    def __init__(self, layer_dims):
        super(InnerEncoder, self).__init__()
        self.encode_hidden_state = tf.keras.Sequential([layers.Dense(l_dim) for l_dim in layer_dims])

    def call(self, inputs):
        x = inputs
        x = self.encode_hidden_state(x)
        return x


class InnerDecoder(layers.Layer):
    def __init__(self, layer_dims):
        super(InnerDecoder, self).__init__()
        self.decode_hidden_state = tf.keras.Sequential([layers.Dense(l_dim) for l_dim in layer_dims])

    def call(self, x):
        # tf.print(x.shape)
        x = self.decode_hidden_state(x)
        return x


class SeqDecoder(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.decoder = InnerDecoder(layer_dims)
        self.repeater = layers.RepeatVector(max_len)
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True)
        # TimeDistributed is better!!!
        self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))
        self.ft_out = layers.TimeDistributed(layers.Dense(ft_len, activation='linear'))

    def call(self, inputs):
        z_sample = inputs
        z_state = self.decoder(z_sample)
        z_input = self.repeater(z_state)
        x = self.lstm_layer(z_input)
        ev_out = self.ev_out(x)
        ft_out = self.ft_out(x)
        return ev_out, ft_out


class SeqProcessEvaluator(metric.JoinedLoss):
    def __init__(self, reduction=losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.edit_distance = metric.MCatEditSimilarity(losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.rec_score = metric.SMAPE(losses.Reduction.SUM_OVER_BATCH_SIZE) # TODO: Fix SMAPE
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        xt_true_events_onehot = utils.to_categorical(true_ev)
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = y_pred
        rec_loss_events = self.edit_distance(true_ev, K.argmax(rec_ev, axis=-1))
        rec_loss_features = self.rec_score(true_ft, rec_ft)
        self._losses_decomposed["edit_distance"] = rec_loss_events
        self._losses_decomposed["feat_mape"] = rec_loss_features

        total = rec_loss_features + rec_loss_events
        return total

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas


class SeqProcessLoss(metric.JoinedLoss):
    def __init__(self, reduction=losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.MSpCatCE(reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)  #.NegativeLogLikelihood(keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = losses.MeanSquaredError(losses.Reduction.SUM_OVER_BATCH_SIZE)  
        self.rec_loss_kl = metric.SimpleKLDivergence(losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        xt_true_events_onehot = utils.to_categorical(true_ev)
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = y_pred
        rec_loss_events = self.rec_loss_events(true_ev, rec_ev)
        rec_loss_features = self.rec_loss_features(true_ft, rec_ft)
        kl_loss = self.rec_loss_kl(z_mean, z_logvar)
        seq_len = tf.cast(tf.shape(true_ev)[-2], tf.float32)
        elbo_loss = (rec_loss_events + rec_loss_features) + (kl_loss * seq_len)  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss_events"] = rec_loss_events
        self._losses_decomposed["rec_loss_features"] = rec_loss_features
        self._losses_decomposed["total"] = elbo_loss

        return elbo_loss

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas