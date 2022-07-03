from __future__ import annotations
from abc import ABC, abstractmethod
import datetime
import pathlib
from pprint import pprint
import time
from typing import Any, Sequence, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from thesis_commons.constants import PATH_RESULTS_MODELS_SPECIFIC

# from tensorflow.keras import Model, layers, optimizers
# from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import Metric, SparseCategoricalAccuracy
from tensorflow.keras import backend as K, losses, metrics, utils, layers, optimizers, models
from thesis_commons.modes import FeatureModes, TaskModeType
from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, EvaluatedCases, SortedCases
from thesis_commons.statististics import StatInstance, StatIteration, StatRow, StatRun
from thesis_viability.viability.viability_function import (MeasureMask, ViabilityMeasure)
import re


class Sampler(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        # Why log(x) - https://stats.stackexchange.com/a/486161
        z_mean, z_log_var = inputs
        # Why log(variance) - https://stats.stackexchange.com/a/486205

        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# class RandomPicker(layers.Layer):
#     def call(self, inputs):
#         z_mean, z_log_var = inputs

#         epsilon = K.random_normal(shape=tf.shape(z_mean))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ReverseEmbedding(layers.Layer):
    def __init__(self, embedding_layer: layers.Embedding, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic)
        self.embedding_layer = embedding_layer

    def call(self, inputs, **kwargs):
        B = self.embedding_layer.get_weights()[0]
        A = K.reshape(inputs, (-1, B.shape[1]))
        similarities = self.cosine_similarity_faf(A, B)
        indices = K.argmax(similarities)
        indices_reshaped = tf.reshape(indices, inputs.shape[:2])
        indices_onehot = utils.to_categorical(indices_reshaped, A.shape[1])

        return indices_onehot

    def cosine_similarity_faf(self, A, B):
        nominator = A @ B
        norm_A = tf.norm(A, axis=1)
        norm_B = tf.norm(B, axis=1)
        denominator = tf.reshape(norm_A, [-1, 1]) * tf.reshape(norm_B, [1, -1])
        return tf.divide(nominator, denominator)


class BaseModelMixin(ConfigurableMixin):
    # def __init__(self) -> None:
    # task_mode_type: TaskModeType = None
    # loss_fn: losses.Loss = None
    # metric_fn: metrics.Metric = None

    def __init__(self, ft_mode: FeatureModes, vocab_len: int, max_len: int, feature_len: int, **kwargs):
        print(__class__)
        super().__init__(**kwargs)
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_len = feature_len
        self.ft_mode = ft_mode
        self._name = type(self).__name__
        self.kwargs = kwargs

    @property
    def name(self):
        return self._name

    def get_config(self) -> BetterDict:
        return BetterDict({'vocab_len': self.vocab_len, 'max_len': self.max_len, 'feature_len': self.feature_len, 'ft_mode': self.ft_mode, 'model': self.name, **self.kwargs})

class GeneratorModelMixin(BaseModelMixin):
    @abstractmethod
    def predict(self, **kwargs):
        return None

class DistanceOptimizerModelMixin(BaseModelMixin):
    def __init__(self, name: str, distance: ViabilityMeasure, *args, **kwargs) -> None:
        print(__class__)
        super(DistanceOptimizerModelMixin, self).__init__(*args, **kwargs)
        self.picks = None
        self.name = name
        self.distance = distance

    def __call__(self, ):
        raise NotImplementedError('Class method needs to be subclassed and overwritten.')

    def predict(self, inputs):
        return self.__call__(inputs)

    def compute_topk_picks(self):
        raise NotImplementedError('Class method needs to be subclassed and overwritten.')

    def compute_viabilities(self, fa_cases: Cases, cf_cases: Cases):
        viability_values = self.distance.compute(fa_cases, cf_cases)
        partial_values = self.distance.partial_values
        return viability_values, partial_values

    def compute_shapes(self, topk, batch_size, seq_len):
        shape_ft = (batch_size, topk, seq_len, -1)
        shape_ev = (batch_size, topk, seq_len)
        shape_viab = (batch_size, topk)
        shape_parts = (-1, batch_size, topk)
        return shape_ev, shape_ft, shape_viab, shape_parts

    def pick_chosen_indices(self, viability_values: np.ndarray, topk: int = 5):
        num_fs, num_cfs = viability_values.shape
        ranking = np.argsort(viability_values, axis=1)
        best_indices = ranking[:, :-topk + 1]
        base_indices = np.repeat(np.arange(num_fs)[..., None], topk, axis=1)

        chosen_indices = np.stack((base_indices.flatten(), best_indices.flatten()), axis=0)
        return chosen_indices, None, ranking

    def pick_topk(self, cf_ev, cf_ft, viabilities, partials, chosen, mask, ranking):
        new_viabilities = viabilities[chosen[0], chosen[1]]
        new_partials = partials[:, chosen[0], chosen[1]]
        chosen_ev, chosen_ft = cf_ev[chosen[1]], cf_ft[chosen[1]]
        return chosen_ev, chosen_ft, new_viabilities, new_partials

    def compute_topk_picks(self, topk, fa_ev, fa_ft, cf_ev, cf_ft):
        batch_size, sequence_length, feature_len = fa_ft.shape
        viab_values, parts_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        chosen, mask, ranking = self.pick_chosen_indices(viab_values, topk)
        shape_ev, shape_ft, shape_viab, shape_parts = self.compute_shapes(topk, batch_size, sequence_length)
        all_shapes = [shape_ev, shape_ft, shape_viab, shape_parts]
        chosen_ev, chosen_ft, new_viabilities, new_partials = self.pick_topk(cf_ev, cf_ft, viab_values, parts_values, chosen, mask, ranking)
        all_picked = [chosen_ev, chosen_ft, new_viabilities, new_partials]
        chosen_ev, chosen_ft, new_viabilities, new_partials = self.compute_reshaping(all_picked, all_shapes)
        picks = {'events': chosen_ev, 'features': chosen_ft, 'viabilities': new_viabilities, 'partials': new_partials}
        return picks, new_partials

    def compute_reshaping(self, all_picked, all_shapes):
        reshaped_picks = tuple([pick.reshape(shape) for pick, shape in zip(all_picked, all_shapes)])
        return reshaped_picks


class ProcessInputLayer(layers.Layer):
    def __init__(self, max_len, feature_len, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.max_len = max_len
        self.feature_len = feature_len
        # self.events = layers.Input(shape=(self.max_len, ), name="events")
        # self.features = layers.Input(shape=(self.max_len, self.feature_len), name="event_features")
        # self.events = layers.InputLayer(input_shape=(self.max_len, ), name="events")
        # self.features = layers.InputLayer(input_shape=(self.max_len, self.feature_len), name="event_features")
        # self.input_layer = layers.InputLayer(shape=[(self.max_len, ),(self.max_len, self.feature_len)], name="event_attributes")

    def build(self, input_shape=None):
        events_shape, features_shape = input_shape
        self.events = layers.InputLayer(input_shape=events_shape[1:], name="events")
        self.features = layers.InputLayer(input_shape=features_shape[1:], name="event_features")

    def call(self, inputs, *args, **kwargs):
        events, features = inputs
        result = self.events(events), self.features(features)
        return result

    # def call(self, inputs, *args, **kwargs):
    #     events, features = inputs
    #     result = [self.events(events), self.features(features)]
    #     # result = self.input_layer(inputs)
    #     # result = events, features
    #     return result

    def get_config(self):
        config = {
            "max_len": self.max_len,
            "feature_len": self.feature_len,
        }
        # config.update(self.kwargs)
        return config


class TensorflowModelMixin(BaseModelMixin, models.Model):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(TensorflowModelMixin, self).__init__(*args, **kwargs)
        # TODO: Turn to layer
        # self.input_layer = layers.InputLayer(input_shape=((self.max_len, ),(self.max_len, self.feature_len)), name="event_attributes")
        self.input_layer = ProcessInputLayer(self.max_len, self.feature_len)

    def build(self, input_shape):
        events_shape, features_shape = input_shape
        # self.events = layers.InputLayer(input_shape=events_shape.shape[1:], name="events")
        # self.features = layers.InputLayer(input_shape=features_shape.shape[1:], name="event_features")
        # self.events = layers.Input(shape=events_shape.shape[1:], name="events")
        # self.features = layers.Input(shape=features_shape.shape[1:], name="event_features")
        # self.input_layer = ProcessInputLayer(self.max_len, self.feature_len)
        self.input_layer.build((events_shape, features_shape))
        self.built = True
        # return super().build(input_shape)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        optimizer = optimizer or optimizers.Adam()
        events_shape, features_shape = (None, self.max_len), (None, self.max_len, self.feature_len)
        self.build((events_shape, features_shape))
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def get_config(self):
        return {**super(BaseModelMixin, self).get_config()}  # Might cause problems with saving

    # def to_dict(self):

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_graph(self) -> models.Model:
        events = layers.Input(shape=(self.max_len, ), name="events")
        features = layers.Input(shape=(self.max_len, self.feature_len), name="event_features")
        results = [events, features]
        summarizer = models.Model(inputs=[results], outputs=self.call(results))
        return summarizer

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        summarizer = self.build_graph()
        summary = summarizer.summary(line_length, positions, print_fn, expand_nested, show_trainable)
        # self.build(summarizer.input_shape[0])
        return summary

    # def call(self, inputs, training=None, mask=None):
    #     result = self.input_layer.call(inputs, training, mask)
    #     return result
    # def call(self, inputs, training=None, mask=None):
    #     return super().call(inputs, training, mask)


class GeneratorWrapper(ConfigurableMixin, ABC):
    

    def __init__(self,
                 predictor: TensorflowModelMixin,
                 generator: BaseModelMixin,
                 evaluator: ViabilityMeasure,
                 measure_mask: MeasureMask = None,
                 top_k: int = None,
                 sample_size: int = None,
                 **kwargs) -> None:
        super(GeneratorWrapper, self).__init__()
        # self.name =
        self.measure_mask = measure_mask or MeasureMask()
        self.evaluator = evaluator
        self.predictor = predictor
        self.generator = generator
        self.run_stats = StatRun()
        self.top_k = top_k
        self.sample_size = sample_size
        self.post_name = ""
        self._config = self.get_config()
        

    def set_measure_mask(self, measure_mask: MeasureMask = None):
        self.measure_mask = measure_mask or MeasureMask()
        return self

    def generate(self, fa_seeds: Cases, **kwargs) -> Sequence[EvaluatedCases]:
        results: Sequence[EvaluatedCases] = []
        pbar = tqdm(enumerate(fa_seeds), total=len(fa_seeds), desc=f"{self.generator.name}") if len(fa_seeds) > 1 else enumerate(fa_seeds)
        self.evaluator = self.evaluator.apply_measure_mask(self.measure_mask)
        stats: StatInstance = None
        for instance_num, fa_case in pbar:
            start_time = time.time()
            # if self.full_name == 'EvoGeneratorWrapper_EvolutionaryStrategy_EvoGeneratorWrapper_DataDistributionSampleInitiator_RouletteWheelSelector_OnePointCrosser_DefaultMutator_BestBreedRecombiner_ImprovementMeasure':
            #     print("STOP model_commons.py")
            generation_results, stats = self.execute_generation(fa_case, **kwargs)
            topk_cases = self.get_topk(generation_results, top_k=self.top_k)
            # if topk_cases == None:
            #     print("STOP model_commons.py")
            #     topk_cases = self.get_topk(generation_results, top_k=self.top_k)
            reduced_results = topk_cases.set_instance_num(instance_num).set_creator(self.generator.name).set_fa_case(fa_case)
            results.append(reduced_results)
            duration = time.time() - start_time
            duration_time = datetime.timedelta(seconds=duration)
            self.run_stats.append(stats.attach('duration', str(duration_time)).attach('duration_s', duration))
        self.construct_model_stats()
        return results

    @abstractmethod
    def execute_generation(self, fc_case, **kwargs) -> Tuple[EvaluatedCases, StatInstance]:
        pass

    @abstractmethod
    def construct_instance_stats(self, info: Any, **kwargs) -> StatInstance:
        pass

    def construct_model_stats(self, **kwargs) -> None:
        self.run_stats.attach('hparams', self.get_config())

    @abstractmethod
    def construct_result(self, generation_results, **kwargs) -> EvaluatedCases:
        pass

    def get_topk(self, result: EvaluatedCases, top_k: int = None) -> SortedCases:
        if top_k is not None:
            return result.get_topk(top_k)
        return result.sort()

    def save_statistics(self, specific_path: pathlib.Path = None, file_name: str = None) -> pathlib.Path:
        try:
            data = self.run_stats.data
            target = (specific_path or PATH_RESULTS_MODELS_SPECIFIC) / ((file_name or self.short_name) + ".csv")
            data.to_csv(target.open("w"), index=False, line_terminator='\n')
            return target
        except Exception as e:
            print(f"SAVING WENT WRONG!!! {e}")
            raise e

    def set_extra_name(self, **kwargs) -> GeneratorWrapper:
        if len(kwargs):
            self.post_name = "_".join([f"{k}{v}" for k, v in kwargs.items()])
        return self

    def construct_instance_stats(self, info: Any, **kwargs) -> StatInstance:
        counterfactual_cases: EvaluatedCases = kwargs.get('counterfactual_cases')
        factual_case: EvaluatedCases = kwargs.get('factual_case')
        
        instance_stats: StatInstance = kwargs.get('stat_instance') or StatInstance()
        iter_stats: StatIteration = kwargs.get('stat_iteration') or StatIteration()
        case: EvaluatedCases = None
        for case in counterfactual_cases:
            stats_row = StatRow()
            stats_row.attach('factual_outcome', factual_case.outcomes[0][0])
            stats_row.attach('target_outcome', ~factual_case.outcomes[0][0])
            stats_row.attach('predicted_outcome', case.outcomes[0][0])
            stats_row.attach('prediction_score', case.viabilities.mllh[0][0])
            stats_row.attach('similarity', case.viabilities.similarity[0][0])
            stats_row.attach('sparcity', case.viabilities.sparcity[0][0])
            stats_row.attach('dllh', case.viabilities.dllh[0][0])
            stats_row.attach('delta', case.viabilities.ollh[0][0])
            stats_row.attach('viability', case.viabilities.viabs[0][0])
            stats_row.attach('events', case.events[0])
            iter_stats.append(stats_row)

        iter_stats.attach(f"n_results", len(counterfactual_cases))
        iter_stats.attach(f"avg_viability", counterfactual_cases.avg_viability[0])
        iter_stats.attach(f"avg_viability", counterfactual_cases.avg_viability[0])
        iter_stats.attach(f"median_viability", counterfactual_cases.median_viability[0])
        iter_stats.attach(f"max_viability", counterfactual_cases.max_viability[0])
        iter_stats.attach(f"min_viability", counterfactual_cases.min_viability[0])

        instance_stats.append(iter_stats)
        return instance_stats

    @property
    def config(self):
        return self._config

    @property
    def full_name(self):
        all_cnfs = self.config
        name_components = "_".join([v for _, _, v in all_cnfs.search('type')])
        name_full = self.name + "_" + self.generator.name + "_" + name_components
        return f"{name_full}"

    @property
    def short_name(self):
        all_cnfs = self.config
        name_components = "_".join([v for _, _, v in all_cnfs.search('type')])
        name_full = self.name + "_" + self.generator.name + "_" + name_components
        name_full = re.sub(r"([a-z])+", "", name_full)
        if (self.post_name is not None) or (self.post_name != ""):
            name_full += "_" + self.post_name
        return name_full

    @property
    def name(self):
        return type(self).__name__

    def get_config(self):
        if not hasattr(self, '_config'):
            self._config = BetterDict(super().get_config()).merge({"wrapper": {'type': self.name}, "gen": self.generator.get_config(), "viab": self.evaluator.get_config()})
            return self._config
        return self._config


