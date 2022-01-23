import tensorflow as tf

from thesis_predictors.helper.evaluation import Evaluator
from ..helper.runner import Runner
from ..helper.metrics import CrossEntropyLoss, CrossEntropyLossModified, MaskedSpCatAcc, MaskedSpCatCE
from ..models.direct_data_lstm import FullLSTMModelOneWayExtensive, FullLSTMModelOneWaySimple
from ..models.lstm import TokenToSequenceLSTM, HybridToSequenceLSTM, SimpleLSTMModelTwoWay
from ..models.seq2seq_lstm import SeqToSeqSimpleLSTMModelOneWay
from ..models.transformer import Seq2SeqTransformerModelOneWay, TransformerModelOneWaySimple, TransformerModelTwoWay
from thesis_readers.helper.modes import FeatureModes, TaskModes
from thesis_readers import DomesticDeclarationsLogReader
from thesis_predictors.helper.constants import EVAL_RESULTS_FOLDER, MODEL_FOLDER

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT
    reader = DomesticDeclarationsLogReader(debug=False, mode=task_mode)
    reader = reader.init_data()
    evaluator = Evaluator(reader).set_task_mode(task_mode)
    results_folder = EVAL_RESULTS_FOLDER
    build_folder = MODEL_FOLDER
    prefix = "test"
    epochs = 1
    batch_size = 32
    adam_init = 0.001
    num_train = 1000
    num_val = 100
    num_test = 1000
    loss_fn = MaskedSpCatCE()
    metric = MaskedSpCatAcc()
    

    r1 = Runner(
        reader,
        FullLSTMModelOneWaySimple(reader.vocab_len, reader.max_len, reader.feature_len - 1),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.FULL_SEP, # Make it part of the model
    ).train_model().evaluate(evaluator, results_folder, prefix)
    r1.save_model(build_folder, prefix)
    r3 = Runner(
        reader,
        HybridToSequenceLSTM(reader.vocab_len, reader.max_len, reader.feature_len),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.EVENT_ONLY,
    ).train_model().evaluate(evaluator, results_folder, prefix)
    r3.save_model(build_folder, prefix)
    r5 = Runner(
        reader,
        TransformerModelOneWaySimple(reader.vocab_len, reader.max_len),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.EVENT_ONLY,
    ).train_model().evaluate(evaluator, results_folder, prefix)
    r5.save_model(build_folder, prefix)