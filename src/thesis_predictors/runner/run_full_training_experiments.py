
# from ..models.lstms.lstm import SimpleLSTM as PredictionModel 
# from ..models.lstms.lstm import BaseLSTM as PredictionModel
from thesis_commons.modes import FeatureModes, TaskModes
from thesis_predictors.helper.constants import (EVAL_RESULTS_FOLDER,
                                                MODEL_FOLDER)
from thesis_predictors.helper.evaluation import Evaluator
# from thesis_readers.readers.DomesticDeclarationsLogReader import DomesticDeclarationsLogReader as Reader
# from thesis_readers import RequestForPaymentLogReader as Reader
from thesis_readers.readers.MockReader import MockReader as Reader

from ..helper.runner import Runner
from ..models.lstms.lstm import EmbeddingLSTM as PredictionModel

DEBUG = True
if __name__ == "__main__":
    # Parameters
    results_folder = EVAL_RESULTS_FOLDER
    build_folder = MODEL_FOLDER
    prefix = "result_next"
    epochs = 50 if not DEBUG else 2
    batch_size = 8 if not DEBUG else 64
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None

    # Setup Reader and Evaluator
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    reader = Reader(debug=False, mode=task_mode)
    data = reader.init_log(save=True)
    reader = reader.init_meta()
    evaluator = Evaluator(reader)
    # adam_init = 0.1

    r1 = Runner(
        PredictionModel,
        reader,
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.FULL,
    ).train_model().evaluate(evaluator, results_folder, prefix)
    # r1.save_model(build_folder, prefix)
    print("done")
    