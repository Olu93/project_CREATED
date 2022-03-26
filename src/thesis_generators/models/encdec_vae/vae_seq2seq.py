from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, Embedding, Activation, Layer, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from thesis_generators.models.model_commons import HybridEmbedderLayer
# TODO: Fix imports by collecting all commons
from thesis_generators.models.model_commons import EmbedderLayer
from thesis_generators.models.model_commons import CustomInputLayer
from thesis_generators.models.model_commons import MetricVAEMixin, LSTMTokenInputMixin, LSTMVectorInputMixin, LSTMHybridInputMixin
from thesis_generators.models.model_commons import GeneratorModelMixin
import thesis_generators.models.model_commons as commons
from thesis_commons import metric
from thesis_predictors.models.model_commons import HybridInput, VectorInput
from typing import Generic, TypeVar, NewType

# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265


class SimpleGeneratorModel(commons.GeneratorPartMixin):

    def __init__(self, ff_dim, layer_dims=[13, 8, 5], *args, **kwargs):
        print(__class__)
        super(SimpleGeneratorModel, self).__init__(*args, **kwargs)
        self.in_layer: CustomInputLayer = None
        self.ff_dim = ff_dim
        layer_dims = [kwargs.get("feature_len") + kwargs.get("embed_dim")] + layer_dims
        self.encoder_layer_dims = layer_dims
        self.encoder = SeqEncoder(self.ff_dim, self.encoder_layer_dims)
        self.sampler = commons.Sampler(self.encoder_layer_dims[-1])
        self.decoder = SeqDecoder(layer_dims[0], self.max_len, self.ff_dim, self.decoder_layer_dims)
        self.embedder = HybridEmbedderLayer(*args, **kwargs)


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = metric.ELBOLoss(name="elbo")
        # metrics = []
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs, training=None, mask=None):
        evs, fts = inputs
        
        z_mean, z_logvar = self.encoder([evs, fts])
        z_sample = self.sampler([z_mean, z_logvar])
        x_evs, x_fts = self.decoder(z_sample)
        return x_evs, x_fts

    def train_step(self, data):
        (events_input, features_input), (events_target, features_target) = data
        metrics_collector = {}
        # Train the Generator.
        with tf.GradientTape() as tape:
            x = self.embedder([events_input, features_input])  # TODO: Dont forget embedding training!!!
            tra_params, inf_params, emi_ev_probs, emi_ft_params = self.generator(x)
            vars = (tra_params, inf_params, emi_ev_probs, emi_ft_params)
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
        
        # TODO: Think of outsourcing this towards a trained inferencer module
        # TODO: It might make sense to introduce a binary sampler and a gaussian sampler
        # TODO: split_params Should be a general utility function instead of a class function. Using it quite often.
        # ev_params = MultiTrainer.split_params(emi_ev_params) 
        # ev_samples = self.sampler(ev_params)
        ft_params = MultiTrainer.split_params(emi_ft_params)
        ft_samples = self.sampler(ft_params) 
        
        eval_loss = self.custom_eval(data[0], (K.argmax(emi_ev_probs), ft_samples))
        if tf.math.is_nan(eval_loss).numpy() or tf.math.is_inf(eval_loss).numpy(): 
            print("We have some trouble here")
        trainer_losses = self.custom_loss.composites
        sanity_losses = self.custom_eval.composites
        losses= {}
        if DEBUG_SHOW_ALL_METRICS:
            losses.update(trainer_losses)
        losses.update(sanity_losses)
        return losses

class SeqEncoder(Model):

    def __init__(self, ff_dim, layer_dims):
        super(SeqEncoder, self).__init__()
        self.lstm_layer = layers.LSTM(ff_dim, return_state=True)
        self.combiner = layers.Concatenate()
        self.encoder = InnerEncoder(layer_dims)
        self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean")
        self.latent_log_var = layers.Dense(layer_dims[-1], name="z_log_var")

    def call(self, inputs):
        x, h, c = self.lstm_layer(inputs)
        x = self.combiner([x, h, c])
        x = self.encoder(x)
        z_mean = self.latent_mean(x)
        z_log_var = self.latent_log_var(x)
        return z_mean, z_log_var


class InnerEncoder(Layer):

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


class SeqDecoder(Model):

    def __init__(self, in_dim, max_len, ff_dim, layer_dims):
        super(SeqDecoder, self).__init__()
        self.max_len = max_len
        self.decoder = InnerDecoder(layer_dims)
        self.repeater = layers.RepeatVector(max_len)
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True)
        self.mean_layer = layers.TimeDistributed(layers.Dense(in_dim))
        self.var_layer = layers.TimeDistributed(layers.Dense(in_dim))

    def call(self, inputs):
        z_sample = inputs
        z_state = self.decoder(z_sample)
        z_input = self.repeater(z_state)
        # x = tf.expand_dims(x,1)
        # z_expanded = tf.repeat(tf.expand_dims(z, 1), self.max_len, axis=1)
        x = self.lstm_layer(z_input)
        x_mean = self.mean_layer(x)
        x_logvar = self.mean_layer(x)
        return x_mean, x_logvar
