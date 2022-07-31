from typing import List, Tuple
import tensorflow as tf

from thesis_commons import metric
from thesis_commons.constants import PATH_MODELS_GENERATORS, PATH_READERS, REDUCTION
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_readers.readers.AbstractProcessLogReader import FeatureInformation
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
        rec_loss_events = self.edit_distance(xt_true_events, ev_samples)
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
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.rec_loss_events = metric.MaskedLoss(losses.SparseCategoricalCrossentropy(reduction=REDUCTION.NONE))  #.NegativeLogLikelihood(keras.REDUCTION.SUM_OVER_BATCH_SIZE)
        self.rec_loss_features = metric.MaskedLoss(losses.MeanSquaredError(reduction=REDUCTION.NONE))
        self.kl_loss = metric.GeneralKLDivergence(reduction=REDUCTION.NONE)
        self.sampler = commons.Sampler()

    def call(self, y_true, y_pred):
        xt_true_events, xt_true_features = y_true
        # xt_true_events_onehot = utils.to_categorical(xt_true_events)
        zt_tra_params, zt_inf_params, xt_emi_ev_probs, zt_emi_ft_params = y_pred
        y_argmax_true, y_argmax_pred, padding_mask = self.compute_mask(xt_true_events, xt_emi_ev_probs)

        # ev_params = SeqProcessLoss.split_params(zt_emi_ev_params)
        ft_params = SeqProcessLoss.split_params(zt_emi_ft_params)
        tra_params = SeqProcessLoss.split_params(zt_tra_params)
        inf_params = SeqProcessLoss.split_params(zt_inf_params)
        ev_loss = self.rec_loss_events.call(xt_true_events, xt_emi_ev_probs, padding_mask=padding_mask)
        ft_loss = self.rec_loss_features.call(xt_true_features, self.sampler(ft_params), padding_mask=padding_mask)
        kl_loss = self.kl_loss.call(inf_params, tra_params)
        # ft_loss = self.rec_loss_features.call(xt_true_features, self.sampler(ft_params), padding_mask=padding_mask)
        
        ev_loss = K.sum(ev_loss, -1)
        ft_loss = K.sum(ft_loss, -1)
        kl_loss = K.sum(kl_loss, -1)        
        elbo_loss = (ev_loss + ft_loss) + kl_loss  # We want to minimize kl_loss and negative log likelihood of q
        self._losses_decomposed["kl_loss"] = kl_loss
        self._losses_decomposed["rec_loss_events"] = ev_loss
        self._losses_decomposed["rec_loss_features"] = ft_loss
        self._losses_decomposed["total"] = elbo_loss
        if any([tf.math.is_nan(l).numpy().any() for k, l in self._losses_decomposed.items()]) or any([tf.math.is_inf(l).numpy().any() for k, l in self._losses_decomposed.items()]):
            print(f"Something happened! - There's at least one nan or inf value")
            ev_loss = self.rec_loss_events(xt_true_events, xt_emi_ev_probs)
            ft_loss = self.rec_loss_features(xt_true_features, self.sampler(ft_params))
            kl_loss = self.kl_loss(inf_params, tra_params)
            elbo_loss = ev_loss + ft_loss - kl_loss
        return elbo_loss

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
        self.future_encoder = FutureSeqEncoder(self.ff_dim)
        self.state_transitioner = TransitionModel(self.ff_dim)
        self.inferencer = InferenceModel(self.ff_dim)
        self.sampler = commons.Sampler()
        self.emitter_events = EmissionEvModel(self.vocab_len)
        self.emitter_features = EmissionFtModel(self.feature_len)
        self.combiner = layers.Concatenate()
        self.masker = layers.Masking()
        self.idxs_discrete = tuple(self.feature_info.idx_discrete.values())
        self.idxs_continuous = tuple(self.feature_info.idx_continuous.values())
        self.custom_loss, self.custom_eval = self.init_metrics(self.idxs_discrete, self.idxs_continuous)


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        # loss = metric.SeqELBOLoss()
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        sampled_z_tra_mean_list = []
        sampled_z_tra_logvar_list = []
        sampled_z_inf_mean_list = []
        sampled_z_inf_logvar_list = []
        sampled_x_probs_list_events = []
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
            xt_emi_ev_probs = self.emitter_events(zt_sample)
            xt_emi_mu_features, xt_emi_logvar_features = self.emitter_features(zt_sample)

            sampled_z_tra_mean_list.append(z_transition_mu)
            sampled_z_tra_logvar_list.append(z_transition_logvar)
            sampled_z_inf_mean_list.append(z_inf_mu)
            sampled_z_inf_logvar_list.append(z_inf_logvar)
            sampled_x_probs_list_events.append(xt_emi_ev_probs)
            sampled_x_emi_mean_list_features.append(xt_emi_mu_features)
            sampled_x_emi_logvar_list_features.append(xt_emi_logvar_features)

        sampled_z_tra_mean = tf.stack(sampled_z_tra_mean_list, axis=1)
        sampled_z_tra_logvar = tf.stack(sampled_z_tra_logvar_list, axis=1)
        sampled_z_inf_mean = tf.stack(sampled_z_inf_mean_list, axis=1)
        sampled_z_inf_logvar = tf.stack(sampled_z_inf_logvar_list, axis=1)
        sampled_x_emi_mean_events = tf.stack(sampled_x_probs_list_events, axis=1)
        sampled_x_emi_mean_features = tf.stack(sampled_x_emi_mean_list_features, axis=1)
        sampled_x_emi_logvar_features = tf.stack(sampled_x_emi_logvar_list_features, axis=1)

        r_tra_params = tf.stack([sampled_z_tra_mean, sampled_z_tra_logvar], axis=-2)
        r_inf_params = tf.stack([sampled_z_inf_mean, sampled_z_inf_logvar], axis=-2)
        r_emi_ev_params = sampled_x_emi_mean_events
        r_emi_ft_params = tf.stack([sampled_x_emi_mean_features, sampled_x_emi_logvar_features], axis=-2)

        return r_tra_params, r_inf_params, r_emi_ev_params, r_emi_ft_params

    def train_step(self, data):
        (events_input, features_input), (events_target, features_target) = data
        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            # x = self.embedder([events_input, features_input])  # TODO: Dont forget embedding training!!!
            tra_params, inf_params, emi_ev_probs, emi_ft_params = self((events_input, features_input), training=True)
            vars = (tra_params, inf_params, emi_ev_probs, emi_ft_params)
            g_loss = self.custom_loss(data[0], vars)
        if tf.math.is_nan(g_loss).numpy().any():
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(g_loss))}")
        if DEBUG_LOSS:
            total_loss = K.sum([val.numpy() for _, val in self.custom_loss.composites.items()])
            composite_losses = {key: val.numpy() for key, val in self.custom_loss.composites.items()}
            print(f"Total loss is {total_loss} with composition {composite_losses}")

        trainable_weights = self.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        # TODO: Think of outsourcing this towards a trained inferencer module
        # TODO: It might make sense to introduce a binary sampler and a gaussian sampler
        # TODO: split_params Should be a general utility function instead of a class function. Using it quite often.
        # ev_params = MultiTrainer.split_params(emi_ev_params)
        # ev_samples = self.sampler(ev_params)
        ft_params = self.split_params(emi_ft_params)
        ft_samples = self.sampler(ft_params)

        eval_loss = self.custom_eval(data[0], (K.argmax(emi_ev_probs), ft_samples))
        if tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy():
            print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_eval.composites
        losses = {}
        if DEBUG_SHOW_ALL_METRICS:
            losses.update(trainer_losses)
        losses.update(sanity_losses)
        return losses

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas

    @staticmethod
    def init_metrics(dscr_cols: List[int], cntn_cols: List[int]) -> Tuple['SeqProcessLoss', 'SeqProcessEvaluator']:
        return [SeqProcessLoss(REDUCTION.NONE), SeqProcessEvaluator()]



# https://youtu.be/rz76gYgxySo?t=1383
class FutureSeqEncoder(models.Model):
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
class TransitionModel(models.Model):
    def __init__(self, ff_dim):
        super(TransitionModel, self).__init__()
        self.hidden = layers.Dense(ff_dim, name="z_tra_hidden", activation='relu')
        # TODO: Centralize this code
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_tra_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_tra_logvar", activation='softplus')

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
        self.hidden = layers.Dense(ff_dim, name="z_inf_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(ff_dim, name="z_inf_mean", activation='linear')
        self.latent_vector_z_log_var = layers.Dense(ff_dim, name="z_inf_logvar", activation='softplus')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(x)
        z_log_var = self.latent_vector_z_log_var(x)
        return z_mean, z_log_var


class EmissionFtModel(models.Model):
    def __init__(self, feature_len):
        super(EmissionFtModel, self).__init__()
        self.hidden = layers.Dense(feature_len, name="x_ft_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(feature_len, name="x_ft_mean", activation=lambda x: 5 * keras.activations.tanh(x))
        self.latent_vector_z_log_var = layers.Dense(feature_len, name="x_ft_logvar", activation='softplus')

    def call(self, inputs):
        z_sample = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(z_sample)
        z_log_var = self.latent_vector_z_log_var(z_sample)
        return z_mean, z_log_var


class EmissionEvModel(models.Model):
    def __init__(self, feature_len):
        super(EmissionEvModel, self).__init__()
        self.hidden = layers.Dense(feature_len, name="x_ev_hidden", activation='relu')
        self.latent_vector_z_mean = layers.Dense(feature_len, name="x_ev", activation='softmax')

    def call(self, inputs):
        z_sample = self.hidden(inputs)
        z_mean = self.latent_vector_z_mean(z_sample)
        return z_mean
    
    
if __name__ == "__main__":
    GModel = DMMModelStepwise
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
    ds_name = "OutcomeBPIC12Reader25"
    reader: AbstractProcessLogReader = AbstractProcessLogReader.load(PATH_READERS / ds_name)
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
