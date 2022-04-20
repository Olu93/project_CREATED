import pathlib
from pydoc import classname
from thesis_commons.libcuts import K, losses, layers, optimizers, models, metrics, utils
import tensorflow as tf
from thesis_commons.functions import sample
from thesis_commons import metric

from typing import Generic, Type, TypeVar, NewType
import thesis_generators.models.model_commons as commons

DEBUG_LOSS = False
DEBUG_SHOW_ALL_METRICS = True


# https://keras.io/examples/generative/conditional_gan/
# TODO: Implement an LSTM version of this
class MultiTrainer(models.Model):

    def __init__(self, GeneratorModel: Type[commons.TensorflowModelMixin], *args, **kwargs):
        super(MultiTrainer, self).__init__(name="_".join([cl.__name__ for cl in [type(self), GeneratorModel]]))
        # Seperately trained
        self.max_len = kwargs.get('max_len')
        self.feature_len = kwargs.get('feature_len')
        self.embed_dim = kwargs.get('embed_dim')
        self.vocab_len = kwargs.get('vocab_len')
        self.in_events = layers.Input(shape=(self.max_len, ))
        self.in_features = layers.Input(shape=(self.max_len, self.feature_len))
        self.sampler = commons.Sampler()
        print("Instantiate generator...")
        self.generator = GeneratorModel(*args, **kwargs)
        self.custom_loss = SeqProcessLoss(losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.custom_eval = SeqProcessEvaluator()

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
        if len(data) == 3:
            (events_input, features_input), (events_target, features_target), sample_weight = data
        else:
            sample_weight = None
            (events_input, features_input), (events_target, features_target) = data

        with tf.GradientTape() as tape:
            vars = self.generator.call([events_input, features_input])  # rec_ev, rec_ft, z_sample, z_mean, z_logvar
            g_loss = self.custom_loss(y_true=[events_target, features_target], y_pred=vars, sample_weight=sample_weight)

            # self.custom_loss(, )
        if tf.math.is_nan(g_loss).numpy():
            print(f"Something happened! - There's at least one nan-value: {K.any(tf.math.is_nan(g_loss))}")
        if DEBUG_LOSS:
            composite_losses = {key: val.numpy() for key, val in self.custom_loss.composites.items()}
            print(f"Total loss is {composite_losses.get('total')} with composition {composite_losses}")

        trainable_weights = self.generator.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        self.generator.optimizer.apply_gradients(zip(grads, trainable_weights))

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

    def summary(self, line_length=None, positions=None, print_fn=None):
        inputs = [self.in_events, self.in_features]
        summarizer = models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        events, features = inputs
        rec_ev, rec_ft, z_sample, z_mean, z_logvar = self.generator.call([events, features])
        return rec_ev, rec_ft
    
    # def save(self, *args, **kwargs):
    #     curr_file_path = pathlib.Path(args[0]) 
    #     args = (str(curr_file_path / "trainer"),) + args[1:]
    #     args_generator = (str(curr_file_path / "generator"),) + args[1:]
        
    #     self.generator.save(*args_generator, **kwargs)
    #     return self.save(*args, **kwargs)
    
    # def save(self, *args, **kwargs):
    #     curr_file_path = pathlib.Path(args[0]) 
    #     args = (str(curr_file_path / "trainer"),) + args[1:]
    #     args_generator = (str(curr_file_path / "generator"),) + args[1:]
        
    #     self.generator.save(*args_generator, **kwargs)
    #     return self.save(*args, **kwargs)
    
    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas
    
    @staticmethod
    def get_loss_and_metrics():
        return [SeqProcessLoss(losses.Reduction.SUM_OVER_BATCH_SIZE), SeqProcessEvaluator()]

    def get_generator(self) -> models.Model:
        return self.generator
    



class SeqProcessEvaluator(metric.JoinedLoss):

    def __init__(self, reduction=losses.Reduction.NONE, name=None, **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.edit_distance = metric.MCatEditSimilarity(losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.rec_score = metric.SMAPE(losses.Reduction.SUM_OVER_BATCH_SIZE)
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
        self.rec_loss_features = losses.MeanSquaredError(losses.Reduction.SUM_OVER_BATCH_SIZE) # TODO: Fix SMAPE
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
        # if any([tf.math.is_nan(l).numpy() for k,l in self._losses_decomposed.items()]) or any([tf.math.is_inf(l).numpy() for k,l in self._losses_decomposed.items()]):
        #     print(f"Something happened! - There's at least one nan or inf value")
        #     rec_loss_events = self.rec_loss_events(true_ev, xt_emi_ev_probs)
        #     rec_loss_features = self.rec_loss_features(true_ft, ft_params)
        #     kl_loss = self.kl_loss(inf_params, tra_params)
        #     elbo_loss = rec_loss_events + rec_loss_features - kl_loss
        return elbo_loss

    @staticmethod
    def split_params(input):
        mus, logsigmas = input[:, :, 0], input[:, :, 1]
        return mus, logsigmas