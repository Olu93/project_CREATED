import tensorflow as tf
from ..helper.runner import Runner
from ..helper.metrics import CrossEntropyLoss, CrossEntropyLossModified, SparseAccuracyMetric, SparseCrossEntropyLoss
from ..models.direct_data_lstm import FullLSTMModelOneWay
from ..models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from ..models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from ..models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from thesis_readers.readers.AbstractProcessLogReader import FeatureModes, TaskModes
from thesis_readers import DomesticDeclarationsLogReader
from thesis_predictors.helper.constants import EVAL_RESULTS_FOLDER, MODEL_FOLDER

if __name__ == "__main__":
    reader = DomesticDeclarationsLogReader(debug=False, mode=TaskModes.OUTCOME)
    # data = data.init_log(save=True)
    reader = reader.init_data()
    results_folder = EVAL_RESULTS_FOLDER
    build_folder = MODEL_FOLDER
    prefix = "test"
    epochs = 2
    batch_size = 32
    adam_init = 0.001
    num_train = 1000
    num_val = 100
    num_test = 1000
    loss_fn = SparseCrossEntropyLoss()
    metric = SparseAccuracyMetric()
    r1 = Runner(
        reader,
        FullLSTMModelOneWay(reader.vocab_len, reader.max_len, reader.feature_len - 1),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.FULL_SEP,
    ).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r1.save_model(build_folder, prefix)
    r3 = Runner(
        reader,
        SimpleLSTMModelOneWay(reader.vocab_len, reader.max_len),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.EVENT_ONLY,
    ).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r3.save_model(build_folder, prefix)
    r5 = Runner(
        reader,
        TransformerModelOneWay(reader.vocab_len, reader.max_len),
        epochs,
        batch_size,
        adam_init,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        ft_mode=FeatureModes.EVENT_ONLY,
    ).train_model(loss_fn, [metric]).evaluate(results_folder, prefix)
    r5.save_model(build_folder, prefix)
