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
        self.rec_loss_events = metric.MSpCatCE(reduction=REDUCTION.SUM_OVER_BATCH_SIZE)  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = losses.MeanSquaredError(REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_kl = metric.SimpleKLDivergence(REDUCTION.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        true_ev, true_ft = y_true
        xt_true_events_onehot = utils.to_categorical(true_ev)
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = y_pred
        rec_loss_events = self.rec_loss_events(true_ev, rec_ev)
        rec_loss_features = self.rec_loss_features(true_ft, rec_ft)
        kl_loss = self.rec_loss_kl(z_mean, z_logvar)
        if DEBUG_LOSS:
            unusual_spike = K.greater_equal(kl_loss, 100)
            any_greater = K.any(unusual_spike)
            if any_greater:
                print(f"We have some trouble here {kl_loss}")
                kl_loss = self.rec_loss_kl(z_mean, z_logvar)
        if DEBUG_LOSS:
            check_nan = tf.math.is_nan([rec_loss_events, rec_loss_features, kl_loss])
            if tf.equal(K.any(check_nan), True):
                print("We have some trouble here")
        seq_len = tf.cast(K.prod(true_ev.shape), tf.float32)
        elbo_loss = (rec_loss_events + rec_loss_features) + kl_loss * seq_len  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss_events"] = rec_loss_events
        self._losses_decomposed["rec_loss_features"] = rec_loss_features
        self._losses_decomposed["total"] = elbo_loss

        return elbo_loss

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "losses": [
                metric.MSpCatCE(reduction=REDUCTION.SUM_OVER_BATCH_SIZE),
                losses.MeanSquaredError(REDUCTION.SUM_OVER_BATCH_SIZE),
                metric.SimpleKLDivergence(REDUCTION.SUM_OVER_BATCH_SIZE)
            ],
            "sampler":
            self.sampler
        })
        return cfg


class SimpleGeneratorModel(commons.TensorflowModelMixin):
    def __init__(self, ff_dim: int, embed_dim: int, layer_dims=[20, 17, 9], mask_zero=0, **kwargs):
        print(__class__)
        super(SimpleGeneratorModel, self).__init__(name=kwargs.pop("name", type(self).__name__), **kwargs)
        # self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        layer_dims = [self.feature_len + embed_dim] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.input_layer = commons.ProcessInputLayer(self.max_len, self.feature_len)
        self.embedder = embedders.EmbedderConstructor(ft_mode=self.ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, max_len=self.max_len, mask_zero=0)
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.sampler = commons.Sampler()
        self.decoder = SeqDecoder(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        # self.decoder = SeqDecoderProbablistic(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)
        self.custom_loss, self.custom_eval = self.init_metrics()
        self.ev_taker = layers.Lambda(lambda x: K.argmax(x))

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super(SimpleGeneratorModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

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
        if (tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy()) and DEBUG_LOSS:
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
        sanity_losses["loss"] = 1 - sanity_losses["edit_distance"] + sanity_losses["feat_mape"]
        losses.update(sanity_losses)
        return losses

    @staticmethod
    def init_metrics() -> Tuple[SeqProcessLoss, SeqProcessEvaluator]:
        return [SeqProcessLoss(REDUCTION.NONE), SeqProcessEvaluator()]

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
        self.lstm_layer = layers.Bidirectional(layers.LSTM(ff_dim, name="lstm_to_bi", activation="tanh", return_sequences=True, dropout=0.5), name="bidirectional_input")
        self.lstm_layer2 = layers.LSTM(ff_dim, return_sequences=False, name="other_lstm", return_state=False, bias_initializer='random_normal', activation='tanh', dropout=0.5)
        self.flatten = layers.Flatten()
        self.norm1 = layers.BatchNormalization()
        # self.repeater = layers.RepeatVector(max_len)
        self.encoder = models.Sequential([layers.Dense(l_dim, activation='softplus') for l_dim in layer_dims])
        # TODO: Maybe add sigmoid or tanh to avoid extremes
        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean", activation="tanh")
        self.latent_log_var = layers.Dense(layer_dims[-1], name="z_logvar", activation="tanh")
        # self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        # self.latent_log_var = layers.Dense(layer_dims[-1], name="z_logvar")

    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = self.lstm_layer2(x)
        x = self.norm1(x)
        # x = self.flatten(x)
        x = self.encoder(x)  # TODO: This converts everything to 0 after 4 steps.
        # print(x[0])
        z_mean = self.latent_mean(x)
        z_logvar = self.latent_log_var(x)
        return z_mean, z_logvar




# https://stackoverflow.com/a/43047615/4162265
class SeqDecoder(models.Model):
    def __init__(self, layer_dims, max_len, ff_dim, vocab_len, ft_len):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.ff_dim = ff_dim
        self.decoder = models.Sequential([layers.Dense(l_dim, activation='softplus') for l_dim in layer_dims])
        self.repeater = layers.RepeatVector(max_len)
        self.condenser = layers.Dense(ff_dim, activation='softmax')
        self.lstm_layer = layers.LSTM(ff_dim,
                                      return_sequences=True,
                                      name="lstm_back_conversion",
                                      return_state=False,
                                      bias_initializer='random_normal',
                                      activation='tanh',
                                      dropout=0.5)
        self.lstm_layer2 = layers.LSTM(ff_dim,
                                       return_sequences=True,
                                       name="lstm_back_conversion2",
                                       return_state=False,
                                       bias_initializer='random_normal',
                                       activation='tanh',
                                       dropout=0.5)
        self.norm1 = layers.BatchNormalization()
        # TimeDistributed is better!!!
        self.ev_out = layers.Dense(vocab_len, activation='softmax')
        self.ft_out = layers.Dense(ft_len, activation='linear')
        # self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))
        # self.ft_out = layers.TimeDistributed(layers.Dense(ft_len, activation='linear'))

    #  https://datascience.stackexchange.com/a/61096/44556
    def call(self, inputs):
        z_sample = inputs
        x = self.decoder(z_sample)
        
        x = self.repeater(x)
        x = self.lstm_layer(x)
        x = self.lstm_layer2(x)
        x = self.norm1(x)
       
        ev_out = self.ev_out(x)
        ft_out = self.ft_out(x)
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
    GModel = SimpleGeneratorModel
    build_folder = PATH_MODELS_GENERATORS
    epochs = 7
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

    train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_target=True)
    val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_target=True)

    model = GModel(ff_dim=ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.num_event_attributes, ft_mode=ft_mode)
    runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init, skip_callbacks=DEBUG_SKIP_SAVING)
    result = model.predict(val_dataset)
    print(result)

# TODO: Fix issue with the OFFSET
# TODO: Check if Offset fits the reconstruction loss
# TODO: Fix val step issue with the fact that it only uses the last always
# TODO: Fix vae returns padding last but for viability we need padding first
