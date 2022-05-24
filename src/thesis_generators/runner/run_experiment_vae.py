from thesis_generators.helper.runner import Runner
from thesis_readers import MockReader as Reader
from thesis_commons.constants import PATH_MODELS_GENERATORS
from thesis_commons.callbacks import CallbackCollection
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_generators.models.encdec_vae.vae_seq2seq import SimpleGeneratorModel as GModel
from thesis_commons.modes import TaskModes

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(20, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)
    val_data = generative_reader.get_dataset(20, DatasetModes.VAL, gen_mode=GeneratorModes.HYBRID, flipped_target=True)


    # TODO: NEEDS BILSTM
