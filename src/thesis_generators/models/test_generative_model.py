import tensorflow as tf
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
    idx = 1
    vocab_len = 21 if reader == None else reader.vocab_len
    max_len = 26 if reader == None else reader.max_len
    current_feature_len = 21 if reader == None else reader.current_feature_len
    training_data = RandomExample(vocab_len, max_len).generate_token_only_generator_example(42)
    model = GeneratorVAETraditional(embed_dim=10, ff_dim=10, vocab_len=vocab_len, max_len=max_len, feature_len=current_feature_len)
    model.compile(run_eagerly=True)
    model(training_data[0])
    model.summary()
    # model.fit(training_data[0], training_data[1])
    train_data = reader.get_dataset_generative(16, DatasetModes.TRAIN)
    val_data = reader.get_dataset_generative(16, DatasetModes.VAL)
    model.fit(train_data, validation_data=val_data, epochs=epochs)
    # tf.stack([tf.cast(tmp[0][:,1], tf.int32),tmp[1]], axis=1)
    print("stuff")
