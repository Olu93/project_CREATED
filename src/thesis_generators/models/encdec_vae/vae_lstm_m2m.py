from typing import Tuple
from thesis_commons.config import DEBUG_EAGER_EXEC, DEBUG_QUICK_TRAIN
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
from thesis_generators.models.encdec_vae.vae_lstm import SimpleLSTMGeneratorModel
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


class SeqProcessLoss(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        self.dscr_cols = tf.constant(kwargs.pop('dscr_cols', []), dtype=tf.int32)  # discrete
        self.cntn_cols = tf.constant(kwargs.pop('cntn_cols', []), dtype=tf.int32)  # continuous
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_ft_discrete = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))
        self.rec_loss_ft_continuous = metric.MaskedLoss(losses.MeanSquaredError(reduction=REDUCTION.NONE))
        self.rec_loss_kl = metric.SimpleKLDivergence(REDUCTION.NONE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        rec_ev, rec_ftd, rec_ftc, z_sample, z_mean, z_logvar = y_pred
        y_argmax_true, y_argmax_pred, padding_mask = self.compute_mask(true_ev, rec_ev)
        # https://stackoverflow.com/a/51139591/4162265
        true_ft_dscr = tf.gather(true_ft, self.dscr_cols, axis=-1)
        true_ft_cntn = tf.gather(true_ft, self.cntn_cols, axis=-1)

        ev_loss = self.rec_loss_events.call(true_ev, rec_ev, padding_mask=padding_mask)
        ft_loss_dscr = self.rec_loss_ft_discrete.call(true_ft_dscr, rec_ftd, padding_mask=padding_mask)
        ft_loss_cntn = self.rec_loss_ft_continuous.call(true_ft_cntn, rec_ftc, padding_mask=padding_mask)
        ft_loss = ft_loss_dscr + ft_loss_cntn
        kl_loss = self.rec_loss_kl(z_mean, z_logvar)
        ev_loss = K.sum(ev_loss, -1)
        ft_loss = K.sum(ft_loss, -1)
        kl_loss = K.sum(kl_loss, -1)
        elbo_loss = ev_loss + ft_loss + kl_loss  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = K.sum(kl_loss)
        # elbo_loss =  K.sum(rec_loss_events + rec_loss_features)
        self._losses_decomposed["rec_loss_events"] = K.sum(ev_loss)
        self._losses_decomposed["rec_loss_features"] = K.sum(ft_loss)
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


class AlignedLSTMGeneratorModel(SimpleLSTMGeneratorModel):
    def __init__(self, ff_dim: int, embed_dim: int, feature_info: FeatureInformation, layer_dims=[20, 17, 9], mask_zero=0, **kwargs):
        super().__init__(ff_dim, embed_dim, feature_info, layer_dims, mask_zero, **kwargs)
        self.encoder = SeqEncoderM2M(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.decoder = SeqDecoderM2M(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        self.l_ev = layers.TimeDistributed(layers.Dense(self.vocab_len, activation='softmax'))
        # self.l_ftd_tmp = layers.TimeDistributed(layers.Dense(3, activation='softmax'))
        self.l_ftd = layers.TimeDistributed(models.Sequential([
            layers.Dense(len(self.idxs_discrete) * 3, activation='linear'),
            layers.Reshape((-1, 3)),
            layers.Activation('softmax'),
        ]))
        self.l_ftc = layers.TimeDistributed(layers.Dense(len(self.idxs_continuous), activation='linear'))
        self.custom_loss, self.custom_eval = self.init_metrics(self.idxs_discrete, self.idxs_continuous)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.embedder(x)
        z_mean, z_logvar = self.encoder(x)
        z_sample = self.sampler([z_mean, z_logvar])
        a, b, c = self.decoder(z_sample)

        x_evs_taken = self.argmaxer(self.l_ev(a))
        x_ftsd_taken = self.argmaxer(self.l_ftd(b))
        x_ftsc_taken = self.l_ftc(c)

        x_ft_tmp = K.concatenate([K.cast(x_ftsd_taken, float), x_ftsc_taken], -1)
        return x_evs_taken, x_ft_tmp

    def train_step(self, data):
        #  TODO: Remove offset from computation
        if len(data) == 3:
            (events_input, features_input), (events_target, features_target), sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), (events_target, features_target) = data

        with tf.GradientTape() as tape:
            x = self.embedder([events_input, features_input])
            z_mean, z_logvar = self.encoder(x)
            z_sample = self.sampler([z_mean, z_logvar])
            x_evs, x_ftds, x_ftcs = self.decoder(z_sample)
            x_evs, x_ftds, x_ftcs = self.l_ev(x_evs), self.l_ftd(x_ftds), self.l_ftc(x_ftcs)
            vars = [x_evs, x_ftds, x_ftcs, z_sample, z_mean, z_logvar]  # rec_ev, rec_ft, z_sample, z_mean, z_logvar
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
        if DEBUG_LOSS and DEBUG_EAGER_EXEC and tf.math.logical_or(tf.math.is_nan(eval_loss), tf.math.is_inf(eval_loss)):
            print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_eval.composites
        losses = {}
        if DEBUG_SHOW_ALL_METRICS:
            losses.update(trainer_losses)
        losses.update(sanity_losses)
        return losses

    @staticmethod
    def init_metrics(dscr_cols: List[int], cntn_cols: List[int]) -> Tuple['SeqProcessLoss', 'SeqProcessEvaluator']:
        return [SeqProcessLoss(REDUCTION.NONE, dscr_cols=dscr_cols, cntn_cols=cntn_cols), SeqProcessEvaluator()]


class SeqEncoderM2M(models.Model):
    def __init__(self, ff_dim, layer_dims, max_len):
        super(SeqEncoderM2M, self).__init__()
        # self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, return_state=True)
        # self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean", activation="linear", bias_initializer='random_uniform')
        # self.latent_logvar = layers.Dense(layer_dims[-1], name="z_logvar", activation="linear", bias_initializer='random_uniform')

        self.encoder = layers.LSTM(ff_dim, return_sequences=False, return_state=False, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.norm1 = layers.BatchNormalization()

        tmp1 = []
        for l_dim in layer_dims:
            tmp1.append(layers.Dense(l_dim, name=f"enc{l_dim}_ev", activation='leaky_relu'))
            tmp1.append(layers.BatchNormalization())

        self.compressor = models.Sequential(tmp1)

        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean_ev", activation='linear')
        self.latent_lvar = layers.Dense(layer_dims[-1], name="z_lvar_ev", activation='linear')

        # self.concat = layers.Concatenate()

    def call(self, inputs):
        x = self.encoder(inputs)
        ev = self.compressor(x)

        z_mean, z_logvar = self.latent_mean(ev), self.latent_lvar(ev)

        return z_mean, z_logvar


class SeqDecoderM2M(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoderM2M, self).__init__()
        self.max_len = max_len
        self.ff_dim = ff_dim
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.Dense(l_dim, activation='leaky_relu'))
            tmp.append(layers.BatchNormalization())
        self.decompressor = models.Sequential(tmp)
        self.lstm_layer = layers.LSTM(layer_dims[-1], return_sequences=True, name="middle", return_state=False, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.lstm_layer_ev = layers.LSTM(ff_dim, return_sequences=True, name="events", return_state=False, activation='tanh', dropout=0.5)
        self.lstm_layer_ft_continuous = layers.LSTM(ff_dim, return_sequences=True, name="features_discrete", return_state=False, activation='tanh', dropout=0.5)
        self.lstm_layer_ft_discrete = layers.LSTM(ff_dim, return_sequences=True, name="features_continuous", return_state=False, activation='tanh', dropout=0.5)
        # self.norm_ev = layers.TimeDistributed(layers.BatchNormalization())
        # self.norm_ft = layers.TimeDistributed(layers.BatchNormalization())
        self.repeat = layers.RepeatVector(self.max_len)

        self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))
        self.ft_discrete = layers.TimeDistributed(layers.Dense(3, activation='softmax'))
        self.ft_continuous = layers.TimeDistributed(layers.Dense(ft_len, activation='linear'))

    #  https://datascience.stackexchange.com/a/61096/44556
    def call(self, inputs):
        z_sample = inputs
        x_decompressed = self.decompressor(z_sample)
        x = self.repeat(x_decompressed)
        h = self.lstm_layer(x)

        a = self.lstm_layer_ev(h)
        b = self.lstm_layer_ft_continuous(h)
        c = self.lstm_layer_ft_discrete(h)
        # a = self.norm_ev(a)
        # b = self.norm_ft(b)
        # ev_out = self.ev_out(a)
        # ftd_out = self.ft_discrete(b)
        # ftc_out = self.ft_continuous(c)
        # return ev_out, ftd_out, ftc_out
        return a, b, c


if __name__ == "__main__":
    GModel = M2MLSTMEncoder
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
    # True false
    train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)
    val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)

    model = GModel(ff_dim=ff_dim,
                   embed_dim=embed_dim,
                   feature_info=reader.feature_info,
                   vocab_len=reader.vocab_len,
                   max_len=reader.max_len,
                   feature_len=reader.feature_len,
                   ft_mode=ft_mode)
    runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init, skip_callbacks=DEBUG_SKIP_SAVING)
    result = model.predict(val_dataset)
    print(result[0])

# TODO: Fix issue with the OFFSET
# TODO: Check if Offset fits the reconstruction loss
# TODO: Fix val step issue with the fact that it only uses the last always
# TODO: Fix vae returns padding last but for viability we need padding first
