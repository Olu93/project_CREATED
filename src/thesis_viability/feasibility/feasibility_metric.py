import io
from math import isnan
from typing import Any, Callable
from unicodedata import is_normalized
import numpy as np
import thesis_viability.helper.base_distances as distances
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes, FeatureModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance
import scipy.stats as stats
import tensorflow as tf
import pickle
from collections import Counter
import pandas as pd

DEBUG = True


# https://gist.github.com/righthandabacus/f1d71945a49e2b30b0915abbee668513
def sliding_window(a, win_size):
    '''Slding window view of a 2D array a using numpy stride tricks.
    For a given input array `a` and the output array `b`, we will have
    `b[i] = a[i:i+w]`
    
    Args:
        a: numpy array of shape (N,M)
    Returns:
        numpy array of shape (K,w,M) where K=N-w+1
    '''
    return np.lib.stride_tricks.sliding_window_view(events, (1, win_size))


class TransitionProbability():

    def __init__(self, events):
        self.events = events
        events_slided = sliding_window(self.events, 2)
        self.transition_counts = pd.DataFrame(events_slided.reshape((-1, 2)).tolist()).value_counts().to_frame()
        self.transition_counts.columns = ['counts']
        self.transition_counts = self.transition_counts.set_index(self.transition_counts.index.set_names(['start', 'end']))
        self.transition_counts_matrix = self.transition_counts['counts'].unstack().fillna(0)
        self.transition_probs_matrix = self.transition_counts_matrix.div(self.transition_counts_matrix.sum(axis=1), axis=0)
        self.transition_probs = self.transition_probs_matrix.melt(ignore_index=False).reset_index()
        self.transition_probs[['start', 'end']] = self.transition_probs[['start', 'end']].astype(int)
        self.transition_probs = self.transition_probs.set_index(['start', 'end'])

        self.start_events = self.events[:, 0]
        self.start_counts = Counter(self.start_events)
        self.starting_probs = {k: cnt / len(events) for k, cnt in self.start_counts.items()}

        # print(self.transition_probs)

    def compute_sequence_probabilities(self, events, is_joint=True):
        events_slided = sliding_window(events, 2)
        events_slided_flat = events_slided.reshape((-1, 2))
        num_pairs, window_size = events_slided_flat.shape
        probs = self.transition_probs.loc[events_slided_flat.tolist()]
        sequential_probabilities = probs.values.reshape(events.shape[0], -1)
        start_events = events[:, 0]
        start_event_prob = np.array([self.starting_probs[ev] for ev in start_events])  # TODO: Optimise by using lookup array
        result = np.hstack([start_event_prob[..., None], sequential_probabilities])
        return result.prod(-1) if is_joint else result

    def compute_sequence_logprobabilities(self, events, is_joint=True):
        sequential_probabilities = self.compute_sequence_probabilities(events, False)
        log_probs = np.log(sequential_probabilities + np.finfo(float).eps)
        stable_joint_log_probs = log_probs.sum(-1)
        return stable_joint_log_probs if is_joint else log_probs

    def compute_cum_probs(self, events, is_log=False):
        sequence_probs = self.compute_sequence_probabilities(events, False).cumprod(-1) if not is_log else self.compute_sequence_logprobabilities(events, False).cumsum(-1)
        return sequence_probs


class EmissionProbability():

    def __init__(self, events, features):
        num_seq, seq_len, num_features = features.shape
        self.events = events
        self.features = features
        events_flat = self.events.reshape((-1, ))
        features_flat = self.features.reshape((-1, num_features))
        sort_indices = events_flat.argsort()
        events_sorted = events_flat[sort_indices]
        features_sorted = features_flat[sort_indices]
        df_ev_and_ft = pd.DataFrame(features_sorted)
        df_ev_and_ft["event"] = events_sorted
        self.gaussian_params = {
            activity: (np.mean(data.drop('event', axis=1).values, axis=0), data.drop('event', axis=1).cov().values, len(data))
            for activity, data in df_ev_and_ft.groupby("event")
        }
        self.gaussian_dists = {k: stats.multivariate_normal(mean=m, cov=c if not np.all(np.isnan(c)) else np.zeros_like(c), allow_singular=True) for k, (m, c, _) in self.gaussian_params.items()}
        print("")

    def compute_probs(self, events, features, is_log=False):
        num_seq, seq_len, num_features = features.shape
        events_flat = events.reshape((-1, ))
        features_flat = features.reshape((-1, num_features))
        unique_events = np.unique(events_flat)
        emission_probs = np.zeros_like(events_flat)
        for ev in unique_events:
            ev_pos = events_flat == ev
            distribution = self.gaussian_dists[ev]
            emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])
            

        result = emission_probs.reshape((num_seq, -1))
        return np.log(result) if is_log else result

class FeasibilityMetric():
    def __init__(self, events, features):
        self.events = events
        self.features = features        
        self.tprobs = TransitionProbability(events)
        self.eprobs = EmissionProbability(events, features)

    def compute_values(self, events, features, is_log=False):
        transition_probs = self.tprobs.compute_cum_probs(events, is_log)
        emission_probs = self.eprobs.compute_probs(events, features, is_log)
        if is_log:
            return transition_probs + emission_probs
        return transition_probs * emission_probs


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    # generative_reader = GenerativeDataset(reader)
    (events, ev_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    metric = FeasibilityMetric(events, ev_features)
    print(metric.compute_values(events, ev_features))

