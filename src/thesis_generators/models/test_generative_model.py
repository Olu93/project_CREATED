import tensorflow as tf
from thesis_generators.models.model_commons import TokenEmbedder
from thesis_generators.models.vae.vae_lstm_adhoc import GeneratorVAETraditional,CustomGeneratorVAE
from thesis_commons.functions import shift_seq_backward
from thesis_readers import DomesticDeclarationsLogReader as Reader
from thesis_commons.modes import TaskModes, FeatureModes


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    reader = Reader(mode=task_mode).init_meta()
    idx = 1
    training_data = reader.get_dataset_generative(ft_mode=FeatureModes.EVENT_ONLY)
    model = GeneratorVAETraditional(embed_dim=10, ff_dim=10, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len)
    model.compile(run_eagerly=True)
    model.summary()
    print("stuff")
    
