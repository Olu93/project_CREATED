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



class AlignedLSTMGeneratorModel(SimpleLSTMGeneratorModel):
    def __init__(self, ff_dim: int, embed_dim: int, feature_info: FeatureInformation, layer_dims=[20, 17, 9], mask_zero=0, **kwargs):
        super().__init__(ff_dim, embed_dim, feature_info, layer_dims, mask_zero, **kwargs)
        self.encoder = SeqEncoderM2M(self.ff_dim, self.encoder_layer_dims, self.max_len)
        self.decoder = SeqDecoderM2M(layer_dims[::-1], self.max_len, self.ff_dim, self.vocab_len, self.feature_len)

class SeqEncoderM2M(models.Model):
    def __init__(self, ff_dim, layer_dims, max_len):
        super(SeqEncoderM2M, self).__init__()
        # self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, return_state=True)
        # self.latent_mean = layers.Dense(layer_dims[-1], name="z_mean", activation="linear", bias_initializer='random_uniform')
        # self.latent_logvar = layers.Dense(layer_dims[-1], name="z_logvar", activation="linear", bias_initializer='random_uniform')

        self.encoder = layers.LSTM(ff_dim, return_sequences=True, return_state=True, bias_initializer='random_uniform', activation='leaky_relu', dropout=0.5, recurrent_dropout=0.5)
        self.norm1 = layers.BatchNormalization()
        
        tmp = []
        for l_dim in layer_dims:
            tmp.append(layers.TimeDistributed(layers.Dense(l_dim, name="z_mean", activation='linear', bias_initializer='random_normal')))
            tmp.append(layers.BatchNormalization())

        self.compressor = models.Sequential(tmp)
        # TODO: Maybe add sigmoid or tanh to avoid extremes
        self.latent_mean = layers.TimeDistributed(layers.Dense(layer_dims[-1], name="z_mean", activation='linear', bias_initializer='random_normal'))
        self.latent_lvar = layers.TimeDistributed(layers.Dense(layer_dims[-1], name="z_lvar", activation='linear', bias_initializer='random_normal'))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.compressor(x)
        z_mean = self.latent_mean(x)
        z_logvar = self.latent_lvar(x)
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
        self.decoder = models.Sequential(tmp)
        self.lstm_layer = layers.LSTM(ff_dim, return_sequences=True, name="middle", return_state=True, bias_initializer='random_uniform', activation='leaky_relu', dropout=0.5, recurrent_dropout=0.5)
        self.lstm_layer_ev = layers.LSTM(ff_dim, return_sequences=True, name="events", return_state=True, bias_initializer='random_uniform', activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.lstm_layer_ft = layers.LSTM(ff_dim, return_sequences=True, name="features", return_state=True, bias_initializer='random_uniform', activation='tanh', dropout=0.5, recurrent_dropout=0.5)
        self.norm_ev = layers.BatchNormalization()
        self.norm_ft = layers.BatchNormalization()
        self.ev_out = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax', bias_initializer='random_normal'))
        self.ft_out = layers.TimeDistributed(layers.Dense(ft_len, activation='linear', bias_initializer='random_normal'))

    #  https://datascience.stackexchange.com/a/61096/44556
    def call(self, inputs):
        z_sample = inputs
        x = self.decoder(z_sample)

        h, h_last, hc_last = self.lstm_layer(x)
        a, a_last, ac_last = self.lstm_layer_ev(h, initial_state=[h_last, hc_last])
        b, b_last, bc_last = self.lstm_layer_ft(h, initial_state=[h_last, hc_last])
        a = self.norm1(a)
        b = self.norm2(b)
        ev_out = self.ev_out(a)
        ft_out = self.ft_out(b)
        return ev_out, ft_out



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
