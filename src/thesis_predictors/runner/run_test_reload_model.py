import tensorflow as tf
from ..helper.runner import Runner
from ..helper.metrics import CrossEntropyLoss, CrossEntropyLossModified, SparseAccuracyMetric, SparseCrossEntropyLoss
from ..models.direct_data_lstm import FullLSTMModelOneWayExtensive
from ..models.lstm import SimpleLSTMModelOneWayExtensive, SimpleLSTMModelTwoWay
from ..models.seq2seq_lstm import SeqToSeqLSTMModelOneWay
from ..models.transformer import TransformerModelOneWayExtensive, TransformerModelTwoWay
from thesis_readers.helper.modes import TaskModes, DatasetModes
from thesis_readers import RequestForPaymentLogReader, VolvoIncidentsReader

if __name__ == "__main__":
    data = VolvoIncidentsReader(debug=False, mode=TaskModes.SIMPLE)
    # data = data.init_log(save=True)
    data = data.init_data()
    results_folder = "results"
    build_folder = "models_bin"
    prefix = "test"
    epochs = 2
    batch_size = 128
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = SparseCrossEntropyLoss()
    metric = SparseAccuracyMetric()

    r5 = Runner(
        data,
        TransformerModelOneWayExtensive(data.vocab_len, data.max_len),
        epochs,
        batch_size,
        adam_init,
        **num_instances,
    ).train_model(loss_fn, [metric])
    # https://keras.io/guides/serialization_and_saving/
    model = tf.keras.models.load_model(r5.save_model(build_folder, prefix).model_path, custom_objects={'SparseCrossEntropyLoss': loss_fn, 'SparseAccuracyMetric': metric})
    print(model.evaluate(data.get_dataset(batch_size, DatasetModes.TEST)))
