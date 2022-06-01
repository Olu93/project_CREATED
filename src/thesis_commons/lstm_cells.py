import tensorflow as tf
import tensorflow_probability as tfp

from thesis_commons.libcuts import (K, layers, losses, metrics, models,
                                    optimizers, utils)
from thesis_commons.model_commons import Sampler

# https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc
# https://stackoverflow.com/questions/39681026/tensorflow-how-to-pass-output-from-previous-time-step-as-input-to-next-timestep


class ProbablisticLSTMCell(layers.Layer):
    
    def __init__(self, num_classes, units=10, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.units = num_classes
        self.prob_estimator = layers.Dense(num_classes, activation="softmax")
        self.sampler = Sampler()
        self.onehot_encoder = layers.Lambda(lambda x: K.one_hot(K.cast(x, tf.uint8), num_classes=num_classes))
        # self.picker = layers.Lambda(lambda x: K.random_normal(shape=tf.shape(z_mean)))
        self.prob_thresholds = layers.Lambda(lambda x: K.cumsum(x))
        self.min_picker = layers.Lambda(lambda x: K.argmin(x))
        self.lstm_cell = layers.LSTMCell(units=units)
        self.concatenator = layers.Concatenate()
        self.masker = layers.Masking()    

    @property
    def state_size(self):
        return self.units, self.units

    def call(self, inputs, states):
        curr_h, curr_c = states
        curr_x = inputs
        input_shape = tf.shape(curr_x)
        seq_len = input_shape[-1]
        batch_size = input_shape[0]
        next_h, next_c = self.lstm_cell(curr_x, states=(curr_h, curr_c))
        input_probs = self.prob_estimator(next_h)
        next_thresholds = self.prob_thresholds(input_probs)
        seed = K.random_normal(shape=input_shape)
        picked = next_thresholds < seed
        index_list = K.repeat(K.arange(seq_len), batch_size)
        index_matrix = tf.reshape(index_list, (batch_size, seq_len))
        mask = self.masker(picked)
        masked_indices_matrix = mask * index_matrix 
        picked_indices = self.min_picker(masked_indices_matrix)
        next_x = picked_indices

        return next_x, (next_h, next_c)

    # def __init__2(self,
    #              num_classes,
    #              units=10,
    #              activation='tanh',
    #              recurrent_activation='sigmoid',
    #              use_bias=True,
    #              kernel_initializer='glorot_uniform',
    #              recurrent_initializer='orthogonal',
    #              bias_initializer='zeros',
    #              unit_forget_bias=True,
    #              kernel_regularizer=None,
    #              recurrent_regularizer=None,
    #              bias_regularizer=None,
    #              kernel_constraint=None,
    #              recurrent_constraint=None,
    #              bias_constraint=None,
    #              dropout=0,
    #              recurrent_dropout=0,
    #              **kwargs):

    #     super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer,
    #                      recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, **kwargs)
    #     self.prob_estimator = layers.Dense(num_classes, activation="softmax")
    #     self.sampler = Sampler()
    #     self.onehot_encoder = layers.Lambda(lambda x: K.one_hot(K.cast(x, tf.uint8), num_classes=num_classes))
    #     # self.picker = layers.Lambda(lambda x: K.random_normal(shape=tf.shape(z_mean)))
    #     self.prob_thresholds = layers.Lambda(lambda x: K.cumsum(x))
    #     self.min_picker = layers.Lambda(lambda x: K.argmin(x))
    #     self.lstm_cell = layers.LSTMCell()
    #     self.concatenator = layers.Concatenate()
    #     self.masker = layers.Masking()



    # def call(self, inputs, states, training=None):
    #     curr_h, curr_c = states
    #     curr_x = inputs
    #     input_shape = tf.shape(curr_x)
    #     seq_len = input_shape[-1]
    #     batch_size = input_shape[0]
    #     next_h, next_c = self.lstm_cell(curr_x, [curr_h, curr_c], training)
    #     input_probs = self.prob_estimator(next_h)
    #     next_thresholds = self.prob_thresholds(input_probs)
    #     seed = K.random_normal(shape=input_shape)
    #     picked = next_thresholds < seed
    #     index_list = K.repeat(K.arange(seq_len), batch_size)
    #     index_matrix = tf.reshape(index_list, (batch_size, seq_len))
    #     mask = self.masker(picked)
    #     masked_indices_matrix = mask * index_matrix 
    #     picked_indices = self.min_picker(masked_indices_matrix)
    #     next_x = picked_indices

    #     return next_x, (next_h, next_c)

    #  Two ways - 1: Pick based on prob disstribution 2: Randomise hidden state and use that


class ProbablisticLSTMCellV2(layers.LSTMCell):
    def __init__(self, num_classes, units=10, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, **kwargs):
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, **kwargs)
        self.prob_estimator = layers.Dense(num_classes, softmax=True)
        self.sampler = Sampler()
        self.onehot_encoder = layers.Lambda(lambda x: K.one_hot(K.cast(x, tf.uint8), num_classes=num_classes))
        self.picker = layers.Lambda(lambda x: K.argmax(x))
        self.concatenator = layers.Concatenate()

    def call(self, inputs, states, training=None):
        curr_h, curr_c = states
        curr_x = inputs

        next_h, next_c = super().call(curr_x, [curr_h, curr_c], training)
        next_h_sampled = self.sampler(curr_x)
        # input_probs = self.prob_estimator(next_h_sampled)

        return next_h_sampled, (next_h_sampled, next_c)