from typing import List, Tuple
import tensorflow as tf

from thesis_commons import metric
from thesis_commons.constants import PATH_MODELS_GENERATORS, PATH_READERS, REDUCTION
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_readers.readers.AbstractProcessLogReader import FeatureInformation
from thesis_readers import OutcomeDice4ELReader

keras = tf.keras
from keras import backend as K, losses, metrics, utils, layers, optimizers, models
import thesis_commons.embedders as embedders

import thesis_commons.model_commons as commons
# TODO: Fix imports by collecting all commons
from thesis_generators.helper.runner import Runner as GRunner

DEBUG_LOSS = False
DEBUG_SHOW_ALL_METRICS = False
DEBUG_QUICK_TRAIN = True
DEBUG_SKIP_SAVING = True
# https://stats.stackexchange.com/a/577483/361976


class SeqProcessEvaluator(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.edit_distance = metric.MCatEditSimilarity(REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_score = metric.SMAPE(REDUCTION.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        xt_true_events, xt_true_features = y_true
        xt_true_events_onehot = utils.to_categorical(xt_true_events)
        ev_samples, ft_samples = y_pred
        rec_loss_events = self.edit_distance(xt_true_events, K.argmax(ev_samples))
        rec_loss_features = self.rec_score(xt_true_features, ft_samples)
        self._losses_decomposed["edit_distance"] = rec_loss_events
        self._losses_decomposed["feat_mape"] = rec_loss_features

        total = rec_loss_features + rec_loss_events
        return total

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas


class SeqProcessLoss(metric.JoinedLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        self.dscr_cols = tf.constant(kwargs.pop('dscr_cols', []), dtype=tf.int32)  # discrete
        self.cntn_cols = tf.constant(kwargs.pop('cntn_cols', []), dtype=tf.int32)  # continuous
        super().__init__(reduction=reduction, name=name, **kwargs)
        # self.rec_loss_events = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        # self.rec_loss_ft_num = metric.MaskedLoss(losses.MeanSquaredError(reduction=REDUCTION.NONE))
        # self.rec_loss_ft_cat = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))
        self.rec_loss_events = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))
        self.rec_loss_ft_num = metric.MaskedLoss(losses.MeanSquaredError(reduction=REDUCTION.NONE))
        self.rec_loss_ft_cat = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))
        # self.kl_loss = losses.KLDivergence(reduction=REDUCTION.SUM_OVER_BATCH_SIZE)
        self.kl_loss = metric.GeneralKLDivergence(reduction=REDUCTION.SUM_OVER_BATCH_SIZE)
        self.sampler = commons.Sampler()
        self.has_nans = tf.constant([float('NaN'), 1.])

    def call(self, y_true, y_pred):
        yt_ev, yt_ft = y_true
        yt_ft_cat = tf.gather(yt_ft, self.dscr_cols, axis=-1)
        yt_ft_num = tf.gather(yt_ft, self.cntn_cols, axis=-1)
        xt_true_events_onehot = utils.to_categorical(yt_ev)
        zt_tra_params, zt_inf_params, zt_emi_params, xt_ev, xt_ft_num, xt_ft_cat = y_pred
        y_argmax_true, y_argmax_pred, padding_mask = self.compute_mask(yt_ev, xt_ev)
        padding_mask = K.ones_like(padding_mask)
        # ev_params = SeqProcessLoss.split_params(zt_emi_ev_params)
        emi_params = SeqProcessLoss.split_params(zt_emi_params)
        tra_params = SeqProcessLoss.split_params(zt_tra_params)
        inf_params = SeqProcessLoss.split_params(zt_inf_params)
        ev_loss = self.rec_loss_events.call(yt_ev, xt_ev, padding_mask=padding_mask)
        ft_loss_num = self.rec_loss_ft_num.call(yt_ft_num, xt_ft_num, padding_mask=padding_mask)
        ft_loss_cat = K.sum(self.rec_loss_ft_cat.call(yt_ft_cat, xt_ft_cat, padding_mask=padding_mask), -1)
        kl_loss = self.kl_loss.call(inf_params, tra_params)
        # ev_loss = tf.select(tf.is_nan(ev_loss), tf.ones_like(ev_loss) * tf.shape(ev_loss)[-1], ev_loss)
        # ft_loss_num = tf.select(tf.is_nan(ft_loss_num), tf.ones_like(ft_loss_num) * tf.shape(ft_loss_num)[-1], ft_loss_num)
        # ft_loss_cat = tf.select(tf.is_nan(ft_loss_cat), tf.ones_like(ft_loss_cat) * tf.shape(ft_loss_cat)[-1], ft_loss_cat)
        # kl_loss = tf.select(tf.is_nan(kl_loss), tf.ones_like(kl_loss) * tf.shape(kl_loss)[-1], kl_loss)
        elbo_loss = (ev_loss + ft_loss_num + ft_loss_cat) #+ kl_loss  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = K.sum(kl_loss, -1)
        self._losses_decomposed["rec_loss_events"] = K.sum(ev_loss, -1)
        self._losses_decomposed["rec_loss_features"] = K.sum(ft_loss_num + ft_loss_cat, -1)
        self._losses_decomposed["total"] = K.sum(elbo_loss, -1)
        if any([tf.math.is_nan(l).numpy().any() for k, l in self._losses_decomposed.items()]) or any([tf.math.is_inf(l).numpy().any() for k, l in self._losses_decomposed.items()]):
            print(f"Something happened! - There's at least one nan or inf value")
            ev_loss = self.rec_loss_events.call(yt_ev, xt_ev, padding_mask=padding_mask)
            ft_loss_num = self.rec_loss_ft_num.call(yt_ft_num, xt_ft_num, padding_mask=padding_mask)
            ft_loss_cat = K.sum(self.rec_loss_ft_cat.call(yt_ft_cat, xt_ft_cat, padding_mask=padding_mask), -1)
            kl_loss = self.kl_loss.call(inf_params, tra_params)
            elbo_loss = (ev_loss + ft_loss_num + ft_loss_cat) + kl_loss
        y_argmax_true, y_argmax_pred, padding_mask = self.compute_mask(yt_ev, xt_ev)
        return K.sum(elbo_loss, -1)#/K.sum(tf.cast(padding_mask, tf.float32), -1, keepdims=True)

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas


class DMMModelStepwise(commons.TensorflowModelMixin):
    def __init__(self, ff_dim, embed_dim, *args, **kwargs):
        print(__class__)
        super(DMMModelStepwise, self).__init__(*args, **kwargs)
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        self.embedder = embedders.EmbedderConstructor(ft_mode=self.ft_mode, vocab_len=self.vocab_len, embed_dim=self.embed_dim, max_len=self.max_len, mask_zero=0)
        # self.feature_len = embed_dim + self.feature_len
        self.initial_z = tf.zeros((1, ff_dim))
        self.is_initial = True
        self.idxs_discrete = tuple(self.feature_info.idx_discrete.values())
        self.idxs_continuous = tuple(self.feature_info.idx_continuous.values())
        self.mask_tmp = list(range(len(self.idxs_discrete) + len(self.idxs_continuous)))
        self.mask_d = tf.cast(tf.constant([[[1 if idx in self.idxs_discrete else 0 for idx in self.mask_tmp]]]), tf.int64)
        self.mask_c = tf.cast(tf.constant([[[1 if idx in self.idxs_continuous else 0 for idx in self.mask_tmp]]]), tf.int64)
        # self.mask_tmp = tf.range(len(self.idxs_discrete) + len(self.idxs_continuous))

        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = commons.Sampler()
        # self.emitter_events = EmissionEvModel(self.vocab_len)
        self.emitter_features = EmissionModel(self.ff_dim)
        self.decoder_ev = DecoderEvModel(self.vocab_len)
        self.decoder_ft = DecoderFtNumModel(K.sum(self.mask_c))
        self.decoder_ft_cat = DecoderFtCatModel(len(self.idxs_discrete), self.max_len)
        self.combiner = layers.Concatenate()
        self.masker = layers.Masking()

        self.custom_loss, self.custom_eval = self.init_metrics(self.idxs_discrete, self.idxs_continuous)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        r_tra_params, r_inf_params, r_emi_params = self.sample_params(inputs, training=None, mask=None)
        x_ev, x_ft_num, x_ft_cat = self.get_predictions(r_emi_params)
        x_ft = K.concatenate([tf.cast(K.argmax(x_ft_cat, axis=-1), tf.float32), x_ft_num], -1)

        return x_ev, x_ft

    def get_predictions(self, r_emi_params):
        x_ev = self.decoder_ev(r_emi_params)
        x_ft_num = self.decoder_ft(r_emi_params)
        x_ft_cat = self.decoder_ft_cat(r_emi_params)
        return x_ev, x_ft_num, x_ft_cat

    def sample_params(self, inputs, training=None, mask=None):
        sampled_z_tra_mean_list = []
        sampled_z_tra_logvar_list = []
        sampled_z_inf_mean_list = []
        sampled_z_inf_logvar_list = []
        # sampled_x_probs_list_events = []
        sampled_x_emi_mean_list_features = []
        sampled_x_emi_logvar_list_features = []
        x = self.input_layer(inputs)
        x = self.embedder(x)
        x = self.future_encoder(x)

        zt_sample = K.zeros_like(x)[:, 0]
        # zt_sample = tf.repeat(self.initial_z, len(gt_backwards), axis=0)
        for t in range(x.shape[1]):
            zt_prev = zt_sample
            xt = x[:, t]

            z_transition_mu, z_transition_logvar = self.state_transitioner(zt_prev)
            z_inf_mu, z_inf_logvar = self.inferencer(self.combiner([xt, zt_prev]))

            zt_sample = self.sampler([z_inf_mu, z_inf_logvar])
            # xt_emi_ev_probs = self.emitter_events(zt_sample)
            z_emi_mu_features, z_emi_logvar_features = self.emitter_features(zt_sample)

            sampled_z_tra_mean_list.append(z_transition_mu)
            sampled_z_tra_logvar_list.append(z_transition_logvar)
            sampled_z_inf_mean_list.append(z_inf_mu)
            sampled_z_inf_logvar_list.append(z_inf_logvar)
            # sampled_x_probs_list_events.append(xt_emi_ev_probs)
            sampled_x_emi_mean_list_features.append(z_emi_mu_features)
            sampled_x_emi_logvar_list_features.append(z_emi_logvar_features)
            # if K.any([K.is_nan(l) for l in [
            #         z_transition_mu,
            #         z_transition_logvar,
            #         z_inf_mu,
            #         z_emi_mu_features,
            #         z_emi_logvar_features,
            # ]]):
            #     print(f"Something happened! - There's at least one nan or inf value")
        sampled_z_tra_mean = tf.stack(sampled_z_tra_mean_list, axis=1)
        sampled_z_tra_logvar = tf.stack(sampled_z_tra_logvar_list, axis=1)
        sampled_z_inf_mean = tf.stack(sampled_z_inf_mean_list, axis=1)
        sampled_z_inf_logvar = tf.stack(sampled_z_inf_logvar_list, axis=1)
        # sampled_x_emi_mean_events = tf.stack(sampled_x_probs_list_events, axis=1)
        sampled_x_emi_mean_features = tf.stack(sampled_x_emi_mean_list_features, axis=1)
        sampled_x_emi_logvar_features = tf.stack(sampled_x_emi_logvar_list_features, axis=1)

        r_tra_params = tf.stack([sampled_z_tra_mean, sampled_z_tra_logvar], axis=-2)
        r_inf_params = tf.stack([sampled_z_inf_mean, sampled_z_inf_logvar], axis=-2)
        # r_emi_ev_params = sampled_x_emi_mean_events
        r_emi_params = tf.stack([sampled_x_emi_mean_features, sampled_x_emi_logvar_features], axis=-2)

        return r_tra_params, r_inf_params, r_emi_params

    def train_step(self, data):
        (events_input, features_input), (events_target, features_target) = data

        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            # x = self.embedder([events_input, features_input])  # TODO: Dont forget embedding training!!!
            tra_params, inf_params, emi_params = self.sample_params((events_input, features_input), training=True)
            x_ev, x_ft_num, x_ft_cat = self.get_predictions(emi_params)
            preds = (tra_params, inf_params, emi_params, x_ev, x_ft_num, x_ft_cat)
            g_loss = self.custom_loss(data[0], preds)
        if tf.math.is_nan(g_loss).numpy().any():
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(g_loss))}")
        if DEBUG_LOSS:
            total_loss = K.sum([val.numpy() for _, val in self.custom_loss.composites.items()])
            composite_losses = {key: val.numpy() for key, val in self.custom_loss.composites.items()}
            print(f"Total loss is {total_loss} with composition {composite_losses}")

        trainable_weights = self.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        # ft_params = self.split_params(emi_params)

        # eval_loss = self.custom_eval(data[0], (x_ev, x_ft))
        # if tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy():
        #     print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        # sanity_losses = self.custom_eval.composites
        losses = {}
        # if DEBUG_SHOW_ALL_METRICS:
        losses.update(trainer_losses)
        # losses.update(sanity_losses)
        return losses

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas

    @staticmethod
    def init_metrics(dscr_cols: List[int], cntn_cols: List[int]) -> Tuple['SeqProcessLoss', 'SeqProcessEvaluator']:
        return [SeqProcessLoss(REDUCTION.NONE, dscr_cols=dscr_cols, cntn_cols=cntn_cols), SeqProcessEvaluator()]


# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(models.Model):
    def __init__(self, ff_dim):
        super(FutureSeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, go_backwards=True)
        # self.combiner = layers.Concatenate()

    def call(self, inputs):
        x = inputs
        x = self.lstm_layer(x)
        # g_t_backwards = self.combiner([h, c])
        return x


# https://youtu.be/rz76gYgxySo?t=1450
class TransitionModel(models.Model):
    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.hidden = layers.Dense(ff_dim, name="z_tra_hidden", activation='sigmoid')
        # TODO: Centralize this code
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_tra_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_tra_logvar", activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)
        return z_mean, z_log_var


# https://youtu.be/rz76gYgxySo?t=1483
class InferenceModel(models.Model):
    def __init__(self, ff_dim):
        super(InferenceModel, self).__init__()
        self.combiner = layers.Concatenate()
        self.hidden = layers.Dense(ff_dim, name="z_inf_hidden", activation='sigmoid')
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_inf_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_inf_logvar", activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)
        return z_mean, z_log_var


class EmissionModel(models.Model):
    def __init__(self, feature_len):
        super(EmissionModel, self).__init__()
        self.hidden = layers.Dense(feature_len, name="x_ft_hidden", activation='sigmoid')
        self.latent_vector_z_mean = layers.Dense(feature_len, name="x_ft_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="x_ft_logvar", activation='linear')

    def call(self, inputs):
        z_sample = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var


class DecoderEvModel(models.Model):
    def __init__(self, vocab_len):
        super(DecoderEvModel, self).__init__()

        # self.hidden = layers.TimeDistributed(layers.Dense(5, activation='relu'))
        # self.out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))
        self.hidden = layers.Dense(5, activation='relu')
        self.out = layers.Dense(vocab_len, activation='softmax')

    def call(self, inputs):
        x = K.concatenate([inputs[:, :, 0], inputs[:, :, 1]])
        x = self.hidden(x)
        x = self.out(x)
        return x


class DecoderFtNumModel(models.Model):
    def __init__(self, feature_len):
        super(DecoderFtNumModel, self).__init__()
        # self.hidden = layers.TimeDistributed(layers.Dense(5, activation='relu'))
        # self.out = layers.TimeDistributed(layers.Dense(feature_len, activation='linear'))
        self.hidden = layers.Dense(5, activation='relu')
        self.out = layers.Dense(feature_len, activation='linear')

    def call(self, inputs):
        x = K.concatenate([inputs[:, :, 0], inputs[:, :, 1]])
        x = self.hidden(x)
        x = self.out(x)
        return x


class DecoderFtCatModel(models.Model):
    def __init__(self, feature_len, max_len):
        super(DecoderFtCatModel, self).__init__()
        self.feature_len = feature_len
        self.max_len = max_len
        # self.hidden = layers.TimeDistributed(layers.Dense(feature_len * 3, activation='relu'))
        self.hidden = layers.Dense(feature_len * 3, activation='relu')
        self.reshape = layers.Reshape((self.max_len, self.feature_len, 3))
        self.out = layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = K.concatenate([inputs[:, :, 0], inputs[:, :, 1]])
        x = self.hidden(x)
        # x = K.reshape(x, (-1, self.max_len, self.feature_len, 3))
        x = self.reshape(x)
        x = self.out(x)
        # bsize, max_len, dim = tf.shape(x)
        return x


if __name__ == "__main__":
    GModel = DMMModelStepwise
    build_folder = PATH_MODELS_GENERATORS
    epochs = 20
    batch_size = 128
    ff_dim = 10 if not DEBUG_QUICK_TRAIN else 10
    embed_dim = 9 if not DEBUG_QUICK_TRAIN else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    task_mode = TaskModes.OUTCOME_PREDEFINED
    ds_name = "OutcomeDice4ELReader"
    reader: AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
    # True false
    train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)
    val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)
    test_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TEST, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)

    model = GModel(ff_dim=ff_dim,
                   embed_dim=embed_dim,
                   feature_info=reader.feature_info,
                   vocab_len=reader.vocab_len,
                   max_len=reader.max_len,
                   feature_len=reader.feature_len,
                   ft_mode=ft_mode)
    runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init, skip_callbacks=DEBUG_SKIP_SAVING).evaluate(test_dataset)
    # result = model.predict(val_dataset)
    # print(result[0])
