
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
from thesis_generators.models.encdec_vae.vae_lstm import SimpleLSTMGeneratorModel
from thesis_generators.models.encdec_vae.vae_lstm_m2m import AlignedLSTMGeneratorModel
from thesis_generators.helper.runner import Runner as GRunner
from thesis_predictors.helper.runner import Runner as PRunner
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PModel
from thesis_readers.helper.helper import get_all_data
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
import pathlib

DEBUG_QUICK_TRAIN = True
if __name__ == "__main__":
    build_folder = PATH_MODELS_GENERATORS
    epochs = 1 if DEBUG_QUICK_TRAIN else 5
    batch_size = 64 if DEBUG_QUICK_TRAIN else 32
    ff_dim = 3 if DEBUG_QUICK_TRAIN else 5
    embed_dim = 4 if DEBUG_QUICK_TRAIN else 9
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL 
    task_mode = TaskModes.OUTCOME_PREDEFINED

    ALL_DATASETS = [ds for ds in ALL_DATASETS if ("Sepsis" in ds) or  ("Dice" in ds)][:1]
    for ds in ALL_DATASETS:
        print(f"\n -------------- Train Generators for {ds} -------------- \n\n")
        try:
            reader: AbstractProcessLogReader = Reader.load(PATH_READERS / ds)
            train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            test_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TEST, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            
            lstm1_name = ds.replace('Reader', '') + AlignedLSTMGeneratorModel.__name__
            model1 = AlignedLSTMGeneratorModel(name=lstm1_name, ff_dim = ff_dim, embed_dim=embed_dim, feature_info=reader.feature_info, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode,)
            runner = GRunner(model1, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)

            print(f"Test Loading of {model1.name}")
            x_ev, x_ft, y_ev, y_ft = reader.gather_full_dataset(test_dataset)
            model1 = models.load_model(PATH_MODELS_GENERATORS/lstm1_name, compile=False)
            y_pred_ev, y_pred_ft = model1.predict((x_ev, x_ft))
            print(f"Load model was successful")
            print("INPUT")
            print(x_ev[0])
            print("PRED")
            print(y_pred_ev[0])
            print("TRUE")
            print(y_ev[0])

            
            # lstm2_name = ds.replace('Reader', 'Generators') + "AlignedLSTMGeneratorModel"
            # model2 = AlignedLSTMGeneratorModel(name=lstm2_name, ff_dim = ff_dim, embed_dim=embed_dim, feature_info=reader.feature_info, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode,)
            # runner = GRunner(model2, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)
        except Exception as e:
            print(f"Something went wrong for {ds} -- {e}")
            print(traceback.format_exc())            
    print("done")