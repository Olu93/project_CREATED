
import traceback
import tensorflow as tf
keras = tf.keras
from keras import models
import numpy as np
from thesis_readers import Reader
from thesis_commons.config import DEBUG_SKIP_DYNAMICS, DEBUG_SKIP_VIZ, DEBUG_USE_MOCK, DEBUG_QUICK_TRAIN, READER
from thesis_commons.constants import (ALL_DATASETS, PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS, PATH_READERS)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_generators.models.encdec_vae.vae_lstm import \
    SimpleLSTMGeneratorModel as GModel
from thesis_generators.helper.runner import Runner as GRunner
from thesis_predictors.helper.runner import Runner as PRunner
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PModel
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
import pathlib
import visualkeras


if __name__ == "__main__":
    build_folder = PATH_MODELS_PREDICTORS
    epochs = 2 if DEBUG_QUICK_TRAIN else 5
    batch_size = 64 if DEBUG_QUICK_TRAIN else 24
    ff_dim = 3 if DEBUG_QUICK_TRAIN else 10
    embed_dim = 4 if DEBUG_QUICK_TRAIN else 9
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    # task_mode = TaskModes.OUTCOME_PREDEFINED
    # ALL_DATASETS = [ds for ds in ALL_DATASETS if ("Sepsis" in ds) or  ("Dice" in ds)]
    # ALL_DATASETS = [ds for ds in ALL_DATASETS if ("Dice" in ds)]
    ALL_DATASETS = [ds for ds in ALL_DATASETS if ("OutcomeTrafficShortReader" in ds)]
    for ds in ALL_DATASETS:
        print(f"\n -------------- Train Predictor for {ds} -------------- \n\n")
        try:
            reader: AbstractProcessLogReader = Reader.load(PATH_READERS / ds)
            train_dataset = reader.get_dataset(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size)
            val_dataset = reader.get_dataset(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size)
            test_dataset = reader.get_test_dataset(ds_mode=DatasetModes.TEST, ft_mode=ft_mode)
            pname = ds.replace('Reader', 'Predictor')
            model1 = PModel(name=pname, ff_dim = ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode)
            runner = PRunner(model1, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)
            print(f"Test Loading of {model1.name}")
            X, y_true = test_dataset
            model1 = models.load_model(PATH_MODELS_PREDICTORS/pname, compile=False)
            y_pred = model1.predict(X)
            print(f"Load model was successful - Acc: {np.mean(y_true==(y_pred>=0.5)*1)}")
        except Exception as e:
            print(f"Something went wrong for {ds} -- {e}")
            print(traceback.format_exc())
    print("done")
    
    # build_folder = PATH_MODELS_GENERATORS
    # epochs = 1 if not DEBUG_QUICK_TRAIN else 2
    # batch_size = 10 if not DEBUG_QUICK_TRAIN else 64
    # ff_dim = 10 if not DEBUG_QUICK_TRAIN else 3
    # embed_dim = 9 if not DEBUG_QUICK_TRAIN else 4
    # adam_init = 0.1
    # num_train = None
    # num_val = None
    # num_test = None
    # ft_mode = FeatureModes.FULL 
    
    # task_mode = TaskModes.OUTCOME_PREDEFINED
    # reader: AbstractProcessLogReader = Reader.load(PATH_READERS / READER)
    
    # train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)
    # val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=True)

    # model = GModel(ff_dim = ff_dim, embed_dim=embed_dim, feature_info=reader.feature_info, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode)
    # runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    # print("done")