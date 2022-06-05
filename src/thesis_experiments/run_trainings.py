

from thesis_commons.config import DEBUG_USE_MOCK, DEBUG_USE_QUICK_MODE, Reader
from thesis_commons.constants import (PATH_MODELS_GENERATORS,
                                      PATH_MODELS_PREDICTORS)
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_generators.helper.runner import Runner as GRunner
from thesis_generators.models.encdec_vae.vae_seq2seq import \
    SimpleGeneratorModel as GModel
from thesis_predictors.helper.runner import Runner as PRunner
from thesis_predictors.models.lstms.lstm import OutcomeLSTM as PModel



if __name__ == "__main__":
    build_folder = PATH_MODELS_PREDICTORS
    epochs = 5 if not DEBUG_USE_QUICK_MODE else 2
    batch_size = 10 if not DEBUG_USE_QUICK_MODE else 64
    ff_dim = 10 if not DEBUG_USE_QUICK_MODE else 3
    embed_dim = 9 if not DEBUG_USE_QUICK_MODE else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(debug=False, mode=task_mode).init_meta(skip_dynamics=True).init_log(save=True)

    train_dataset = reader.get_dataset(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size)
    val_dataset = reader.get_dataset(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size)

    model = PModel(ff_dim = ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
    runner = PRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")
    
    build_folder = PATH_MODELS_GENERATORS
    epochs = 1 if not DEBUG_USE_QUICK_MODE else 2
    batch_size = 10 if not DEBUG_USE_QUICK_MODE else 64
    ff_dim = 10 if not DEBUG_USE_QUICK_MODE else 3
    embed_dim = 9 if not DEBUG_USE_QUICK_MODE else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL 
    
    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(debug=False, mode=task_mode).init_meta(skip_dynamics=True).init_log(save=True)
    
    train_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size,  flipped_target=True)
    val_dataset = reader.get_dataset_generative(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size,  flipped_target=True)

    model = GModel(ff_dim = ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
    runner = GRunner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")