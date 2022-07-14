
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


if __name__ == "__main__":
    build_folder = PATH_MODELS_PREDICTORS
    epochs = 5 if not DEBUG_QUICK_TRAIN else 2
    batch_size = 10 if not DEBUG_QUICK_TRAIN else 64
    ff_dim = 10 if not DEBUG_QUICK_TRAIN else 3
    embed_dim = 9 if not DEBUG_QUICK_TRAIN else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    task_mode = TaskModes.OUTCOME_PREDEFINED

    for ds in ALL_DATASETS:
        print(f"\n -------------- Train Predictor for {ds} -------------- \n\n")
        reader: AbstractProcessLogReader = Reader.load(PATH_READERS / ds)
        train_dataset = reader.get_dataset(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size)
        val_dataset = reader.get_dataset(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size)
        pname = ds.replace('Reader', 'Predictor') + PModel.__class__.__name__
        model = PModel(name=pname, ff_dim = ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode)
        reader:AbstractProcessLogReader = PRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")