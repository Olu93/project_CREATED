
# from thesis_predictors.helper.evaluation import Evaluator
from thesis_commons.constants import PATH_MODELS_PREDICTORS
# from ..models.lstms.lstm import SimpleLSTM as PredictionModel
# from ..models.lstms.lstm import BaseLSTM as PredictionModel
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
# from thesis_readers.readers.DomesticDeclarationsLogReader import DomesticDeclarationsLogReader as Reader
# from thesis_readers import RequestForPaymentLogReader as Reader
from thesis_readers import OutcomeMockReader as Reader

from ..helper.runner import Runner
from ..models.lstms.lstm import OutcomeLSTM as PModel

DEBUG = True
if __name__ == "__main__":
    build_folder = PATH_MODELS_PREDICTORS
    epochs = 5 if not DEBUG else 2
    batch_size = 10 if not DEBUG else 64
    ff_dim = 10 if not DEBUG else 3
    embed_dim = 9 if not DEBUG else 4
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(debug=False, mode=task_mode).init_meta(skip_dynamics=True).init_log(save=True)

    train_dataset = reader.get_dataset(ds_mode=DatasetModes.TRAIN, ft_mode=ft_mode, batch_size=batch_size)
    val_dataset = reader.get_dataset(ds_mode=DatasetModes.VAL, ft_mode=ft_mode, batch_size=batch_size)

    model = PModel(ff_dim = ff_dim, embed_dim=embed_dim, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len, ft_mode=ft_mode)
    runner = Runner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")
