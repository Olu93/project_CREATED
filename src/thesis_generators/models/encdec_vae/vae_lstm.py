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


# TODO: Fixes for nan vals https://stackoverflow.com/a/37242531/4162265
class SeqProcessLoss(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        self.dscr_cols = tf.constant(kwargs.pop('dscr_cols', []), dtype=tf.int32) # discrete
        self.cntn_cols = tf.constant(kwargs.pop('cntn_cols', []), dtype=tf.int32) # continuous
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = metric.MaskedLoss(losses.MeanSquaredError(reduction=REDUCTION.NONE))
        self.rec_loss_kl = metric.SimpleKLDivergence(REDUCTION.NONE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = y_pred
        y_argmax_true, y_argmax_pred, padding_mask = self.compute_mask(true_ev, rec_ev)
        # https://stackoverflow.com/a/51139591/4162265
        true_ft_dscr, rec_ft_dscr = tf.gather(true_ft, self.dscr_cols, axis=-1), tf.gather(rec_ft, self.dscr_cols, axis=-1) 
        true_ft_cntn, rec_ft_cntn = tf.gather(true_ft, self.cntn_cols, axis=-1), tf.gather(rec_ft, self.cntn_cols, axis=-1) 

        ev_loss = self.rec_loss_events.call(true_ev, rec_ev, padding_mask=padding_mask)
        ft_loss_dscr = self.rec_loss_features.call(true_ft_dscr, rec_ft_dscr, padding_mask=padding_mask)
        ft_loss_cntn = self.rec_loss_features.call(true_ft_cntn, rec_ft_cntn, padding_mask=padding_mask)
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


class SimpleLSTMGeneratorModel(commons.TensorflowModelMixin):
    def __init__(self, ff_dim: int, embed_dim: int, feature_info: FeatureInformation, layer_dims=[20, 17, 9], mask_zero=0, **kwargs):
        super(SimpleLSTMGeneratorModel, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        # self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        self.feature_info = feature_info
        layer_dims = [self.feature_len + embed_dim] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.input_layer = commons.ProcessInputLayer(self.max_len, self.feature_len)
        self.embedder = embedders.EmbedderConstructor(ft_mode=self.ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, max_len=self.max_len, mask_zero=0)
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.sampler = commons.Sampler()
        self.decoder = SeqDecoder(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        # self.decoder = SeqDecoderProbablistic(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        self.idxs_discrete = list(self.feature_info.idx_discrete.values())
        self.idxs_continuous = list(self.feature_info.idx_discrete.values())
        self.custom_loss, self.custom_eval = self.init_metrics(self.idxs_discrete, self.idxs_continuous)
        self.ev_taker = layers.Lambda(lambda x: K.argmax(x))

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super(SimpleLSTMGeneratorModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.embedder(x)
        z_mean, z_logvar = self.encoder(x)
        z_sample = self.sampler([z_mean, z_logvar])
        x_evs, x_fts = self.decoder(z_sample)
        x_evs_taken = self.ev_taker(x_evs)
        return x_evs_taken, x_fts

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
        if DEBUG_LOSS and DEBUG_EAGER_EXEC and tf.math.logical_or(tf.math.is_nan(eval_loss) , tf.math.is_inf(eval_loss)):
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
    #     vars = [x_evs, x_fts, z_sample, z_mean, z_logvar]
    #     eval_loss = self.custom_eval(data[1], vars)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     losses = {}
    #     sanity_losses = self.custom_eval.composites
    #     sanity_losses["loss"] = 1 - sanity_losses["edit_distance"] + sanity_losses["feat_mape"]
    #     losses.update(sanity_losses)
    #     return losses

    @staticmethod
    def init_metrics(dscr_cols: List[int], cntn_cols: List[int]) -> Tuple['SeqProcessLoss', 'SeqProcessEvaluator']:
        return [SeqProcessLoss(REDUCTION.NONE, dscr_cols=dscr_cols, cntn_cols=cntn_cols), SeqProcessEvaluator()]

    def get_config(self):
        return {self.custom_loss.name: self.custom_loss, self.custom_eval.name: self.custom_eval}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# TODO: Fix issue with waaaay to large Z's
class SeqEncoder(models.Model):
    def __init__(self, ff_dim, layer_dims, max_len):
        super(SeqEncoder, self).__init__()
        # self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, return_state=True)
        # self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean", activation="linear", bias_initializer='random_uniform')
        # self.latent_logvar = layers.Dense(layer_dims[-1], name="z_logvar", activation="linear", bias_initializer='random_uniform')

        self.flatten = layers.Flatten()
        self.norm1 = layers.BatchNormalization()
        # self.repeater = layers.RepeatVector(max_len)
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.Dense(l_dim, activation='leaky_relu'))
            tmp.append(layers.BatchNormalization())

        self.encoder = models.Sequential(tmp)
        # TODO: Maybe add sigmoid or tanh to avoid extremes
        self.lstm_layer = layers.Bidirectional(layers.LSTM(layer_dims[-1], name="enc_start", return_sequences=True, return_state=False, bias_initializer='random_uniform', activation='tanh', dropout=0.5, recurrent_dropout=0.5), merge_mode='mul')
        # self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        # self.latent_lvar = layers.Dense(layer_dims[-1], name="z_lvar")
        self.latent_mean = layers.TimeDistributed(layers.Dense(layer_dims[-1], name="z_mean", activation='linear'))
        self.latent_lvar = layers.TimeDistributed(layers.Dense(layer_dims[-1], name="z_lvar", activation='linear'))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.lstm_layer(x)
        z_mean = self.latent_mean(x)
        z_logvar = self.latent_lvar(x)
        return z_mean, z_logvar


class SeqDecoder(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.ff_dim = ff_dim
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.Dense(l_dim, activation='leaky_relu'))
            tmp.append(layers.BatchNormalization())
        self.decoder = models.Sequential(tmp)
        self.repeater = layers.RepeatVector(max_len)
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim, return_sequences=True, name="middle", return_state=True, bias_initializer='random_uniform', activation='leaky_relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='mul')
        self.lstm_layer_ev = layers.LSTM(ff_dim, return_sequences=True, name="events", return_state=False, bias_initializer='random_uniform', activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.lstm_layer_ft = layers.LSTM(ff_dim, return_sequences=True, name="features", return_state=False, bias_initializer='random_uniform', activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
        # self.flatten = layers.Flatten()
        # self.mixer = layers.Dense(layer_dims[-1], activation='leaky_relu', bias_initializer='random_normal')
        # self.mixer2 = layers.Dense(layer_dims[-1], activation='leaky_relu', bias_initializer='random_normal')
        # TimeDistributed is better!!!
        # self.ev_out = layers.Dense(vocab_len, activation='softmax', bias_initializer='random_normal')
        # self.ft_out = layers.Dense(ft_len, activation='linear', bias_initializer='random_normal')
        self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax', bias_initializer='random_normal'))
        self.ft_out = layers.TimeDistributed(layers.Dense(ft_len, activation='linear', bias_initializer='random_normal'))

    #  https://datascience.stackexchange.com/a/61096/44556
    def call(self, inputs):
        z_sample = inputs
        x = self.decoder(z_sample)
        # x = self.repeater(x)
        # batch = tf.shape(x)[0]
        # x = self.norm1(x)
        # x = self.flatten(x)
        # x = self.mixer(x)
        # x = self.mixer2(x)
        # x = K.reshape(x, (batch, self.max_len, -1))
        h, h_last_fw, hc_last_fw, h_last_bw, hc_last_bw = self.lstm_layer(x)
        h_last = h_last_fw * h_last_bw
        hc_last = hc_last_fw * hc_last_bw
        a = self.lstm_layer_ev(h, initial_state=[h_last, hc_last])
        b = self.lstm_layer_ft(h, initial_state=[h_last, hc_last])
        a = self.norm1(a)
        b = self.norm2(b)
        ev_out = self.ev_out(a)
        ft_out = self.ft_out(b)
        return ev_out, ft_out


class SeqDecoderProbablistic(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoderProbablistic, self).__init__()
        self.max_len = max_len
        self.decoder = models.Sequential([layers.Dense(l_dim, activation='softplus') for l_dim in layer_dims])
        # self.lstm_layer = layers.RNN(ProbablisticLSTMCell(vocab_len), return_sequences=True, name="lstm_probablistic_back_conversion")
        self.lstm_cell = ProbablisticLSTMCell(vocab_len)
        # TimeDistributed is better!!!
        # self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))
        self.ft_out = layers.TimeDistributed(layers.Dense(ft_len, activation='linear'))

    def call(self, inputs):
        z_sample = inputs
        z_state = self.decoder(z_sample)
        x = z_state
        state = K.zeros_like(z_state), K.zeros_like(z_state)
        state_collector = []
        x_collector = []
        for i in range(self.max_len):
            x, state = self.lstm_cell(x, state)
            state_collector.append(state[0])
            x_collector.append(x)
        ev_out = tf.stack(x_collector)
        h_out = tf.stack(state_collector, axis=1)
        # x = self.lstm_layer(zeros, initial_state=z_state)
        # ev_out = self.ev_out(x)
        ft_out = self.ft_out(h_out)
        return ev_out, ft_out


if __name__ == "__main__":
    GModel = SimpleLSTMGeneratorModel
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
