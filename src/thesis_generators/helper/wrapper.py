import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from thesis_commons import metric
from thesis_commons.functions import reverse_sequence_2
from thesis_commons.modes import DatasetModes, FeatureModes, GeneratorModes
from thesis_predictors.helper.constants import MODEL_FOLDER
from thesis_readers import AbstractProcessLogReader


class ModelWrapper():
    prediction_model: Model = None

    def __init__(self, reader: AbstractProcessLogReader, model_num: int = None, model_name: str = None) -> None:
        self.reader = reader
        self.model_dirs = {x.name: x for x in MODEL_FOLDER.iterdir() if x.is_dir()}
        if model_num:
            self.load_model_by_num(model_num)
        if model_name:
            self.load_model_by_name(model_name)

    def load_model_by_path(self, model_path: pathlib.Path):
        self.model_path = model_path
        self.prediction_model = keras.models.load_model(self.model_path,
                                                        custom_objects={
                                                            'MaskedSpCatCE': metric.MaskedSpCatCE(),
                                                            'MaskedSpCatAcc': metric.MaskedSpCatAcc(),
                                                            'MaskedEditSimilarity': metric.MaskedEditSimilarity(),
                                                        })
        self.model_name = self.prediction_model.name
        return self

    def load_model_by_num(self, model_num: int):
        self.model_num = model_num
        chosen_model = list(self.model_dirs.items())[self.model_num][1]
        return self.load_model_by_path(chosen_model)

    def load_model_by_name(self, model_name: str):
        chosen_model = self.model_dirs[model_name]
        return self.load_model_by_path(chosen_model)

    def prepare_input(self, example):
        structure = self.reader.get_dataset()._structure[0]

        shape_batch = (example.shape[0], )
        return (
            tf.constant(example, dtype=tf.float32),
            tf.zeros(shape_batch + structure[1].shape[1:]),
            tf.zeros(shape_batch + structure[2].shape[1:]),
            tf.zeros(shape_batch + structure[3].shape[1:]),
        )

    def predict_sequence(self, sequence) -> np.ndarray:
        sequence = sequence[None] if sequence.ndim < 2 else sequence
        input_for_prediction = self.prepare_input(sequence)
        return self.prediction_model.predict(input_for_prediction)


class GenerativeDataset():
    def __init__(self, reader: AbstractProcessLogReader) -> None:
        self.reader = reader
        self.vocab_len = reader.vocab_len
        self.max_len = reader.max_len
        self.current_feature_len = reader.feature_len



    def get_dataset(self,
                    batch_size=1,
                    data_mode: DatasetModes = DatasetModes.TRAIN,
                    ft_mode: FeatureModes = FeatureModes.FULL,
                    gen_mode: GeneratorModes = GeneratorModes.HYBRID, flipped_target=False, is_weighted = False):
        results = None
        features, _ = self.reader._choose_dataset_shard(data_mode)
        res_features = self.reader._prepare_input_data(features, ft_mode=FeatureModes.FULL)[0]
        
        sample_weights = self.reader._compute_sample_weights(res_features[0])
        flipped_res_features = (reverse_sequence_2(res_features[0]), reverse_sequence_2(res_features[1]))
        if is_weighted:
            results = (res_features, flipped_res_features if flipped_target else res_features, sample_weights)
        else: 
            results = (res_features, flipped_res_features if flipped_target else res_features)
            
        self.current_feature_len = res_features[1].shape[-1]
        return tf.data.Dataset.from_tensor_slices(results).batch(batch_size)
