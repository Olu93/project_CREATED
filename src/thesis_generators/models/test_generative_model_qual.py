import tensorflow as tf
from thesis_commons.metrics import VAELoss
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes
from thesis_commons.example_data import RandomExample
from thesis_generators.models.model_commons import TokenEmbedderLayer
from thesis_generators.models.vae.vae_lstm_adhoc import GeneratorVAETraditional, CustomGeneratorVAE
from thesis_commons.functions import shift_seq_backward
from thesis_readers import DomesticDeclarationsLogReader as Reader
from thesis_commons.modes import TaskModes, FeatureModes

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 20
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(16, DatasetModes.TRAIN)
    val_data = generative_reader.get_dataset(16, DatasetModes.VAL)

    model = GeneratorVAETraditional(embed_dim=10,
                                    ff_dim=10,
                                    vocab_len=generative_reader.vocab_len,
                                    max_len=generative_reader.max_len,
                                    feature_len=generative_reader.current_feature_len)
    
    model.compile(run_eagerly=True)
    x_pred, y_true = next(iter(train_data))
    y_pred = model(x_pred)
    model.summary()
    # model.fit(training_data[0], training_data[1])
    # loss_fn = VAELoss()
    # loss = loss_fn(y_true, y_pred)
    model.fit(train_data, validation_data=val_data, epochs=epochs)
    # tf.stack([tf.cast(tmp[0][:,1], tf.int32),tmp[1]], axis=1)
    print("stuff")
    # TODO: NEEDS BILSTM
