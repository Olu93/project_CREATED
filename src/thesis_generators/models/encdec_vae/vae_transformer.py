from typing import Tuple
from thesis_commons.config import DEBUG_QUICK_TRAIN
from thesis_commons.constants import PATH_MODELS_GENERATORS, PATH_MODELS_PREDICTORS, REDUCTION

import tensorflow as tf

keras = tf.keras
from keras import backend as K, layers, losses, models, utils, optimizers

import thesis_commons.embedders as embedders
# TODO: Fix imports by collecting all commons
import thesis_commons.model_commons as commons
from thesis_commons import metric
from thesis_commons.callbacks import CallbackCollection
from thesis_commons.constants import PATH_MODELS_GENERATORS
from thesis_commons.lstm_cells import ProbablisticLSTMCell
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.readers.AbstractProcessLogReader import \
    AbstractProcessLogReader
from thesis_readers import *
from thesis_readers.helper.helper import get_all_data
from thesis_readers import Reader
from thesis_generators.helper.runner import Runner as GRunner
from thesis_predictors.helper.runner import Runner as PRunner
# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True
DEBUG_SKIP_SAVING = True


class SeqProcessEvaluator(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.edit_distance = metric.MCatEditSimilarity(REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_score = metric.SMAPE(REDUCTION.SUM_OVER_BATCH_SIZE)  # TODO: Fix SMAPE
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

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "losses": [
                metric.MCatEditSimilarity(REDUCTION.SUM_OVER_BATCH_SIZE),
                metric.SMAPE(REDUCTION.SUM_OVER_BATCH_SIZE),
            ],
            "sampler": self.sampler
        })
        return cfg


# TODO: Fixes for nan vals https://stackoverflow.com/a/37242531/4162265
class SeqProcessLoss(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.MSpCatCE(reduction=REDUCTION.NONE)  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = losses.MeanSquaredError(REDUCTION.NONE)
        self.rec_loss_kl = metric.SimpleKLDivergence(REDUCTION.NONE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        seq_len = tf.cast(K.prod(true_ev.shape), tf.float32)
        xt_true_events_onehot = utils.to_categorical(true_ev)
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = y_pred
        rec_loss_events = self.rec_loss_events(true_ev, rec_ev)
        rec_loss_features = self.rec_loss_features(true_ft, rec_ft)
        kl_loss = self.rec_loss_kl(z_mean, z_logvar)
        rec_loss_events = K.sum(rec_loss_events, -1)
        rec_loss_features = K.sum(rec_loss_features, -1)
        kl_loss = K.sum(kl_loss, -1)
        elbo_loss = rec_loss_events + rec_loss_features + kl_loss  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = K.sum(kl_loss)
        # elbo_loss =  K.sum(rec_loss_events + rec_loss_features)
        self._losses_decomposed["rec_loss_events"] = K.sum(rec_loss_events)
        self._losses_decomposed["rec_loss_features"] = K.sum(rec_loss_features)
        self._losses_decomposed["total"] = K.sum(elbo_loss)

        return elbo_loss

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "losses": [metric.MSpCatCE(reduction=REDUCTION.NONE),
                       losses.MeanSquaredError(REDUCTION.NONE),
                       metric.SimpleKLDivergence(REDUCTION.NONE)],
            "sampler": self.sampler
        })
        return cfg


class SimpleTransformerModel(commons.TensorflowModelMixin):
    def __init__(self, ff_dim: int, embed_dim: int, layer_dims=[20, 17, 9], mask_zero=0, **kwargs):
        print(__class__)
        super(SimpleTransformerModel, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        # self.in_layer: CustomInputLayer = None
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = 10
        self.pos_dim = 10
        self.rate1 = 0.1
        self.rate2 = 0.1
        self.vector_len = self.feature_len + self.embed_dim + self.pos_dim
        layer_dims = [self.vector_len] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.input_layer = commons.ProcessInputLayer(self.max_len, self.feature_len)
        self.embedder = embedders.EmbedderConstructor(ft_mode=self.ft_mode,
                                                      vocab_len=self.vocab_len,
                                                      embed_dim=self.embed_dim,
                                                      max_len=self.max_len,
                                                      pos_dim=self.pos_dim,
                                                      mask_zero=0)
        self.transformer = TransformerBlock(self.vector_len, self.num_heads, self.ff_dim)

        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.latent_mean = layers.TimeDistributed(layers.Dense(self.vector_len, name="z_mean", activation='linear', bias_initializer='random_normal'))
        self.latent_lvar = layers.TimeDistributed(layers.Dense(self.vector_len, name="z_lvar", activation='linear', bias_initializer='random_normal'))
        self.sampler = commons.Sampler()
        self.decoder = SeqDecoder(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)

        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate1)
        # self.downsampler = layers.TimeDistributed(layers.Dense(self))

        self.ev_out = layers.TimeDistributed(layers.Dense(self.vocab_len, activation='softmax', bias_initializer='random_normal'))
        self.ft_out = layers.TimeDistributed(layers.Dense(self.feature_len, activation='linear', bias_initializer='random_normal'))

        self.custom_loss, self.custom_eval = self.init_metrics()
        # self.custom_loss = losses.MeanSquaredError()
        # self.custom_eval = losses.MeanSquaredError()
        self.ev_taker = layers.Lambda(lambda x: K.argmax(x))

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super(SimpleTransformerModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        events_input, features_input = inputs
        embeddings = self.embedder([events_input, features_input])

        x = self.transformer(embeddings)
        x = self.encoder(x)
        z_mean = self.latent_mean(x)
        z_lvar = self.latent_lvar(x)
        z_sample = self.sampler([z_mean, z_lvar])
        x = self.decoder(z_sample)

        attn_output = self.att(embeddings, embeddings)
        y = self.layernorm1(embeddings + attn_output)

        x = self.transformer(x + y)
        ev_out = self.ev_out(x)
        ft_out = self.ft_out(x)
        return ev_out, ft_out

    def train_step(self, data):
        #  TODO: Remove offset from computation
        if len(data) == 3:
            (events_input, features_input), (events_target, features_target), sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), (events_target, features_target) = data

        with tf.GradientTape() as tape:
            embeddings = self.embedder([events_input, features_input])

            x = self.transformer(embeddings)
            x = self.encoder(x)
            z_mean = self.latent_mean(x)
            z_lvar = self.latent_lvar(x)
            z_sample = self.sampler([z_mean, z_lvar])
            x = self.decoder(z_sample)

            attn_output = self.att(embeddings, embeddings)
            y = self.layernorm1(embeddings + attn_output)

            x = self.transformer(x + y)
            ev_out = self.ev_out(x)
            ft_out = self.ft_out(x)

            vars = [ev_out, ft_out, z_sample, z_mean, z_lvar]  # rec_ev, rec_ft, z_sample, z_mean, z_logvar
            g_loss = self.custom_loss(y_true=[events_target, features_target], y_pred=vars, sample_weight=sample_weight)

        trainable_weights = self.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        eval_loss = self.custom_eval(data[1], vars)
        if (tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy()) and DEBUG_LOSS:
            print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_eval.composites
        losses = {}
        if DEBUG_SHOW_ALL_METRICS:
            losses.update(trainer_losses)
        losses.update(sanity_losses)
        return losses

    # def test_step(self, data):
    #     # Unpack the data
    #     if len(data) == 3:
    #         (events_input, features_input), (events_target, features_target), sample_weight = data
    #     else:
    #         sample_weight = None
    #         (events_input, features_input), (events_target, features_target) = data  # Compute predictions
    #     x = self.embedder([events_input, features_input])
    #     z_mean, z_logvar = self.encoder(x)
    #     z_sample = self.sampler([z_mean, z_logvar])
    #     x_evs, x_fts = self.decoder(z_sample)
    #     vars = [x_evs, x_fts, z_sample, z_mean, z_logvar]  # rec_ev, rec_ft, z_sample, z_mean, z_logvar        # Updates the metrics tracking the loss
    #     eval_loss = self.custom_eval(data[1], vars)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     losses = {}
    #     sanity_losses = self.custom_eval.composites
    #     sanity_losses["loss"] = 1 - sanity_losses["edit_distance"] + sanity_losses["feat_mape"]
    #     losses.update(sanity_losses)
    #     return losses

    @staticmethod
    def init_metrics() -> Tuple['SeqProcessLoss', 'SeqProcessEvaluator']:
        return [SeqProcessLoss(REDUCTION.NONE), SeqProcessEvaluator()]

    def get_config(self):

        return {self.custom_loss.name: self.custom_loss, self.custom_eval.name: self.custom_eval}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        inputs = inputs
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


# TODO: Fix issue with waaaay to large Z's
class SeqEncoder(models.Model):
    def __init__(self, ff_dim, layer_dims, max_len):
        super(SeqEncoder, self).__init__()
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.Dense(l_dim, activation='leaky_relu'))
            tmp.append(layers.BatchNormalization())

        self.encoder = models.Sequential(tmp)

    def call(self, inputs):
        x = self.encoder(inputs)
        return x


class SeqDecoder(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoder, self).__init__()
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.Dense(l_dim, activation='leaky_relu'))
            tmp.append(layers.BatchNormalization())

        self.decoder = models.Sequential(tmp)

    #  https://datascience.stackexchange.com/a/61096/44556
    def call(self, inputs):
        z_sample = inputs
        x = self.decoder(z_sample)
        return x


if __name__ == "__main__":
    GModel = SimpleTransformerModel
    build_folder = PATH_MODELS_GENERATORS
    epochs = 10
    batch_size = 10 if not DEBUG_QUICK_TRAIN else 64
    ff_dim = 10 if not DEBUG_QUICK_TRAIN else 3
    embed_dim = 9 if not DEBUG_QUICK_TRAIN else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader: AbstractProcessLogReader = Reader.load()

    train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=True, flipped_output=True)
    val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=True, flipped_output=True)

    model = GModel(ff_dim=ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.num_event_attributes, ft_mode=ft_mode)
    runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init, skip_callbacks=DEBUG_SKIP_SAVING)
    result = model.predict(val_dataset)
    print(result[0])

# TODO: Fix issue with the OFFSET
# TODO: Check if Offset fits the reconstruction loss
# TODO: Fix val step issue with the fact that it only uses the last always
# TODO: Fix vae returns padding last but for viability we need padding first
