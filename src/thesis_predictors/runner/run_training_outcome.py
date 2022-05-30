import tensorflow as tf
from thesis_commons.modes import DatasetModes
import thesis_commons.model_commons as commons
# from thesis_predictors.helper.evaluation import Evaluator
from thesis_predictors.helper.constants import EVAL_RESULTS_FOLDER, MODEL_FOLDER

# from thesis_readers.readers.DomesticDeclarationsLogReader import DomesticDeclarationsLogReader as Reader
# from thesis_readers import RequestForPaymentLogReader as Reader
from thesis_readers import OutcomeMockReader as Reader
from ..helper.runner import Runner
from ..models.lstms.lstm import OutcomeLSTM as PModel
# from ..models.lstms.lstm import SimpleLSTM as PredictionModel
# from ..models.lstms.lstm import BaseLSTM as PredictionModel
from thesis_commons.modes import FeatureModes, TaskModes

DEBUG = True
if __name__ == "__main__":
    # Parameters
    results_folder = EVAL_RESULTS_FOLDER
    build_folder = MODEL_FOLDER
    prefix = "result_next"
    epochs = 5 if not DEBUG else 2
    batch_size = 10 if not DEBUG else 64
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL

    # Setup Reader and Evaluator
    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(debug=False, mode=task_mode)
    data = reader.init_log(save=True)
    reader = reader.init_meta(skip_dynamics=True)

    train_dataset = reader.get_dataset(batch_size, DatasetModes.TRAIN, ft_mode=ft_mode)
    val_dataset = reader.get_dataset(batch_size, DatasetModes.VAL, ft_mode=ft_mode)
    test_dataset = reader.get_dataset_with_indices(DatasetModes.TEST, ft_mode=ft_mode)
    if num_train:
        train_dataset = train_dataset.take(num_train)
    if num_val:
        val_dataset = val_dataset.take(num_val)
    if num_test:
        test_dataset = test_dataset.take(num_test)  # TODO: IMPORTANT FIX - Was the wrong parameter!!!!

    model = PModel(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
    r1 = Runner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")
