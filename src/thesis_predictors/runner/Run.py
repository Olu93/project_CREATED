import tensorflow as tf
from thesis_predictors.helper.constants import EVAL_RESULTS_FOLDER, MODEL_FOLDER

from thesis_readers.readers.DomesticDeclarationsLogReader import DomesticDeclarationsLogReader
from ..helper.runner import Runner
from ..helper.metrics import CrossEntropyLoss, CrossEntropyLossModified, SparseAccuracyMetric, SparseCrossEntropyLoss
from ..models.direct_data_lstm import FullLSTMModelOneWay
from ..models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from ..models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from ..models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from thesis_readers.readers.AbstractProcessLogReader import TaskModes
from thesis_readers import RequestForPaymentLogReader

if __name__ == "__main__":
    data = DomesticDeclarationsLogReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    results_folder = EVAL_RESULTS_FOLDER
    build_folder = MODEL_FOLDER
    prefix = "result"
    epochs = 5
    batch_size = 64
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = SparseCrossEntropyLoss()
    metric = SparseAccuracyMetric()
    r1 = Runner(data, FullLSTMModelOneWay(data.vocab_len, data.max_len, data.feature_len - 1), epochs, batch_size, adam_init,
                **num_instances).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r1.save_model(build_folder, prefix)
    r3 = Runner(
        data,
        SimpleLSTMModelOneWay(data.vocab_len, data.max_len),
        epochs,
        batch_size,
        adam_init,
        **num_instances,
    ).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r3.save_model(build_folder, prefix)
    r5 = Runner(
        data,
        TransformerModelOneWay(data.vocab_len, data.max_len),
        epochs,
        batch_size,
        adam_init,
        **num_instances,
    ).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r5.save_model(build_folder, prefix)
