from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_readers import MockReader as Reader
from thesis_commons.constants import PATH_MODELS_GENERATORS
from thesis_commons.callbacks import CallbackCollection
from thesis_generators.models.model_commons import HybridEmbedderLayer
from thesis_generators.models.encdec_vae.joint_trainer import MultiTrainer
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_generators.models.baselines.casebased_heuristic import CaseBasedGeneratorModel as GModel
from thesis_commons.modes import TaskModes
from thesis_commons.constants import PATH_MODELS_PREDICTORS
import tensorflow as tf
import os
import thesis_commons.metric as metric


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL_SEP)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL_SEP)
 
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}    
    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    predictive_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects)
    
    DEBUG = True

    
    model = GModel(
        example_cases=(cf_events, cf_features),
        vocab_len=generative_reader.vocab_len,
        max_len=generative_reader.max_len,
        feature_len=generative_reader.current_feature_len,
    )
    
    model.fit((tr_events, tr_features), predictive_model)

    model.predict((fa_events, fa_features))

    print("stuff")
    # TODO: NEEDS BILSTM
