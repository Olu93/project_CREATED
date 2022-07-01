import os

import tensorflow as tf
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models

import thesis_commons.metric as metric
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.modes import DatasetModes, FeatureModes, TaskModes
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_generators.models.baselines.casebased_heuristic import \
    CaseBasedGenerator as GModel
from thesis_readers import MockReader as Reader


def print_results(fa_events, fa_features, model, index=0, show_features=False, show_partials=False):
    print("Reference Sequence:")
    print(fa_events[index])
    if show_features:
        print("Reference Features:")
        print(fa_features[index])
    print("Computed Counterfactual Events:")
    print(model.picks['events'][index])
    if show_features:
        print("Computed Counterfactual Features:")
        print(model.picks['features'][index])
    print("Computed Viabilities:")
    print(model.picks['viabilities'][index])
    if show_partials:
        print("Computed Partials:")
        for partial, values in model.distance.parts.items():
            print(f'{partial}: {values[index]}')
        

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
    predictive_model = models.load_model(PATH_MODELS_PREDICTORS / all_models[0], custom_objects=custom_objects)
    
    DEBUG = True

    
    model = GModel(
        example_cases=(cf_events, cf_features),
        vocab_len=reader.vocab_len,
        max_len=reader.max_len,
        feature_len=reader.get_event_attr_len(FeatureModes.FULL),
    )
    
    model.fit((tr_events, tr_features), predictive_model)

    top_n_cases = model.predict((fa_events, fa_features))

    print("\n=================")
    print_results(fa_events, fa_features, model, index=0, show_features=False, show_partials=True)
    print("\n=================")
    print_results(fa_events, fa_features, model, index=2, show_features=False, show_partials=True)
    print("Done")
