
import traceback
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

DEBUG_QUICK_TRAIN = False
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

    for ds in ALL_DATASETS[:1]:
        print(f"\n -------------- Train Generators for {ds} -------------- \n\n")
        try:
            reader: AbstractProcessLogReader = Reader.load(PATH_READERS / ds)
            train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            test_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TEST, ft_mode=ft_mode, batch_size=batch_size, flipped_input=False, flipped_output=False)
            
            lstm1_name = ds.replace('Reader', 'Generators') + "SimpleLSTMGeneratorModel"
            model1 = SimpleLSTMGeneratorModel(name=lstm1_name, ff_dim = ff_dim, embed_dim=embed_dim, feature_info=reader.feature_info, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode,)
            runner = GRunner(model1, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)
            PModel.load(lstm1_name)
            # lstm2_name = ds.replace('Reader', 'Generators') + "AlignedLSTMGeneratorModel"
            # model2 = AlignedLSTMGeneratorModel(name=lstm2_name, ff_dim = ff_dim, embed_dim=embed_dim, feature_info=reader.feature_info, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode,)
            # runner = GRunner(model2, reader).train_model(train_dataset, val_dataset, epochs, adam_init).evaluate(test_dataset)
        except Exception as e:
            print(f"Something went wrong for {ds} -- {e}")
            print(traceback.format_exc())            
    print("done")