from thesis_generators.helper.runner import Runner
from thesis_readers import OutcomeMockReader as Reader
from thesis_commons.constants import PATH_MODELS_GENERATORS
from thesis_commons.callbacks import CallbackCollection
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel as GModel
from thesis_commons.modes import TaskModes
from thesis_commons.modes import FeatureModes


DEBUG = True
if __name__ == "__main__":
    build_folder = PATH_MODELS_GENERATORS
    epochs = 5 if not DEBUG else 2
    batch_size = 10 if not DEBUG else 64
    adam_init = 0.1
    num_train = None
    num_val = None
    num_test = None
    ft_mode = FeatureModes.FULL 
    
    task_mode = TaskModes.OUTCOME_PREDEFINED
    reader = Reader(debug=False, mode=task_mode).init_meta(skip_dynamics=True).init_log(save=True)
    
    train_dataset = reader.get_dataset_generative(batch_size, DatasetModes.TRAIN,  flipped_target=True)
    val_dataset = reader.get_dataset_generative(batch_size, DatasetModes.VAL,  flipped_target=True)

    if num_train:
        train_dataset = train_dataset.take(num_train)
    if num_val:
        val_dataset = val_dataset.take(num_val)

    model = GModel(ff_dim = 5, embed_dim=4, vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.current_feature_len, ft_mode=ft_mode)
    runner = Runner(model, reader).train_model(train_dataset, val_dataset, epochs, adam_init)

    print("done")
    
