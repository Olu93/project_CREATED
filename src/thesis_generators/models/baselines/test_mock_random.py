import os

import tensorflow as tf

import thesis_commons.metric as metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_generators.models.baselines.random_search import \
    RandomGenerator as GModel
from thesis_readers import MockReader as Reader

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    (tr_events, tr_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL)
    (fa_events, fa_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=FeatureModes.FULL)
    (cf_events, cf_features), _ = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=FeatureModes.FULL)
 
    custom_objects = {obj.name: obj for obj in [metric.MSpCatCE(), metric.MSpCatAcc(), metric.MEditSimilarity()]}    
    all_models = os.listdir(PATH_MODELS_PREDICTORS)
    predictive_model = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects)
    
    DEBUG = True

    
    model = GModel(
        example_cases=(cf_events, cf_features),
        vocab_len=reader.vocab_len,
        max_len=reader.max_len,
        feature_len=reader.get_event_attr_len(FeatureModes.FULL),
    )
    
    model.fit((tr_events, tr_features), predictive_model)

    top_n_cases = model.predict((fa_events, fa_features))

    print("stuff")
    # TODO: NEEDS BILSTM
    
    print(top_n_cases[0].shape)
