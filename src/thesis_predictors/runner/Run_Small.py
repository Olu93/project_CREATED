import tensorflow as tf
from ..helper.runner import Runner
from ..helper.metrics import CrossEntropyLoss, CrossEntropyLossModified, SparseCrossEntropyLoss
from ..models.direct_data_lstm import FullLSTMModelOneWay
from ..models.lstm import SimpleLSTMModelOneWay, SimpleLSTMModelTwoWay
from ..models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from ..models.transformer import TransformerModelOneWay, TransformerModelTwoWay
from thesis_data_readers.AbstractProcessLogReader import ShapeModes, TaskModes
from thesis_data_readers import RequestForPaymentLogReader

if __name__ == "__main__":
    data = RequestForPaymentLogReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    folder = "results"
    prefix = "test"
    epochs = 10
    batch_size = 32
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": 100, "num_test": 1000}
    # loss_fn = CrossEntropyLoss()
    loss_fn = SparseCrossEntropyLoss()
    # loss_fn_mod = CrossEntropyLossModified()
    r = Runner(data, FullLSTMModelOneWay(data.vocab_len, data.max_len, data.feature_len-1), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
    r = Runner(data, SeqToSeqLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
    r = Runner(data, SimpleLSTMModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
    r = Runner(data, SimpleLSTMModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
    r = Runner(data, TransformerModelOneWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
    r = Runner(data, TransformerModelTwoWay(data.vocab_len, data.max_len), epochs, batch_size, adam_init, **num_instances).train_model(loss_fn).evaluate(folder, prefix)
