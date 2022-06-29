import tensorflow as tf
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models
from thesis_commons.metric import MaskedSpCatAcc, MaskedSpCatCE
from thesis_commons.modes import DatasetModes, TaskModes
from thesis_readers import VolvoIncidentsReader

from ..helper.runner import Runner
from ..models.transformer import (Seq2SeqTransformerModelOneWay,
                                  TransformerModelOneWaySimple)

if __name__ == "__main__":
    results_folder = "results"
    build_folder = "models_bin"
    prefix = "test"
    epochs = 2
    batch_size = 128
    adam_init = 0.001
    num_instances = {"num_train": None, "num_val": None, "num_test": None}
    loss_fn = MaskedSpCatCE()
    metric = MaskedSpCatAcc()

    data = VolvoIncidentsReader(debug=False, mode=TaskModes.NEXT_EVENT_EXTENSIVE)
    data = data.init_meta()
    r5 = Runner(
        data,
        Seq2SeqTransformerModelOneWay(data.vocab_len, data.max_len),
        epochs,
        batch_size,
        adam_init,
        **num_instances,
    ).train_model()
    # https://keras.io/guides/serialization_and_saving/
    model = models.load_model(r5.save_model(build_folder, prefix).model_path, custom_objects={'SparseCrossEntropyLoss': loss_fn, 'SparseAccuracyMetric': metric})
    print(model.evaluate(data.get_dataset(batch_size, DatasetModes.TEST)))
    
    
    data = VolvoIncidentsReader(debug=False, mode=TaskModes.NEXT_EVENT)
    data = data.init_meta()
    r5 = Runner(
        data,
        TransformerModelOneWaySimple(data.vocab_len, data.max_len),
        epochs,
        batch_size,
        adam_init,
        **num_instances,
    ).train_model()
    # https://keras.io/guides/serialization_and_saving/
    model = .models.load_model(r5.save_model(build_folder, prefix).model_path, custom_objects={'SparseCrossEntropyLoss': loss_fn, 'SparseAccuracyMetric': metric})
    print(model.evaluate(data.get_dataset(batch_size, DatasetModes.TEST)))
