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
def sliding_window(events, win_size):
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

    def __init__(self, events, vocab_len):
        self.events = events
        events_slided = sliding_window(self.events, 2)
        self.trans_count_matrix = np.zeros((vocab_len, vocab_len))
        self.trans_probs_matrix = np.zeros((vocab_len, vocab_len))
        
        self.df_tra_counts = pd.DataFrame(events_slided.reshape((-1, 2)).tolist()).value_counts()
        self.trans_idxs = np.array(self.df_tra_counts.index.tolist(), dtype=np.int)
        self.trans_from = self.trans_idxs[:, 0]
        self.trans_to = self.trans_idxs[:, 1]
        self.trans_counts = np.array(self.df_tra_counts.values.tolist(), dtype=np.int)
        self.trans_count_matrix[self.trans_from, self.trans_to] = self.trans_counts
        self.trans_probs_matrix = self.trans_count_matrix / self.trans_count_matrix.sum(axis=1, keepdims=True)
        self.trans_probs_matrix[np.isnan(self.trans_probs_matrix)] = 0


        self.start_count_matrix = np.zeros((vocab_len, 1))
        self.start_events = self.events[:, 0]
        self.start_counts_counter = Counter(self.start_events)
        self.start_indices = np.array(list(self.start_counts_counter.keys()), dtype=np.int)
        self.start_counts = np.array(list(self.start_counts_counter.values()), dtype=np.int)
        self.start_count_matrix[self.start_indices, 0] = self.start_counts
        self.start_probs = self.start_count_matrix/self.start_counts.sum()


    def compute_sequence_probabilities(self, events, is_joint=True):
        events_slided = sliding_window(events, 2)
        events_slided_flat = events_slided.reshape((-1, 2))
        transistions = np.array(events_slided_flat.tolist(), dtype=int)
        t_from = transistions[:, 0]
        t_to = transistions[:, 1]
        probs = self.trans_probs_matrix[t_from, t_to].reshape(events.shape[0], -1)
        start_events =  np.array(list(events[:, 0]), dtype=np.int)
        start_event_prob = self.start_probs[start_events, 0, None]  
        result = np.hstack([start_event_prob, probs])
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
    # TODO: Create class that simplifies the dists to assume feature independence
    def __init__(self, events, features):
        num_seq, seq_len, num_features = features.shape
        self.events = events
        self.features = features
        events_flat = self.events.reshape((-1, ))
        features_flat = self.features.reshape((-1, num_features))
        sort_indices = events_flat.argsort()
        events_sorted = events_flat[sort_indices]
        features_sorted = features_flat[sort_indices]
        self.df_ev_and_ft = pd.DataFrame(features_sorted)
        self.df_ev_and_ft["event"] = events_sorted
        self.estimate_params()

    def estimate_params(self):
        self.gaussian_params = {
            activity: (np.mean(data.drop('event', axis=1).values, axis=0), data.drop('event', axis=1).cov().values, len(data))
            for activity, data in self.df_ev_and_ft.groupby("event")
        }
        self.gaussian_dists = {k: stats.multivariate_normal(mean=m, cov=c if not np.all(np.isnan(c)) else np.zeros_like(c), allow_singular=True) for k, (m, c, _) in self.gaussian_params.items()}

    def compute_probs(self, events, features, is_log=False):
        num_seq, seq_len, num_features = features.shape
        events_flat = events.reshape((-1, ))
        features_flat = features.reshape((-1, num_features))
        unique_events = np.unique(events_flat)
        emission_probs = np.zeros_like(events_flat)
        for ev in unique_events:
            ev_pos = events_flat == ev
            distribution = self.gaussian_dists.get(ev, None)  
            emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos]) if distribution else 0
            # distribution = self.gaussian_dists[ev]
            # emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])            

        result = emission_probs.reshape((num_seq, -1))
        return np.log(result) if is_log else result
    
class EmissionProbabilityIndependentFeatures(EmissionProbability):
    def estimate_params(self):
        self.gaussian_params = {
            activity: (np.mean(data.drop('event', axis=1).values, axis=0), np.diag(data.drop('event', axis=1).cov().values), len(data))
            for activity, data in self.df_ev_and_ft.groupby("event")
        }
        self.gaussian_dists = {k: stats.multivariate_normal(mean=m, cov=c if not np.all(np.isnan(c)) else np.zeros_like(c), allow_singular=True) for k, (m, c, _) in self.gaussian_params.items()}

# TODO: Call it data likelihood or possibility measure 
# TODO: Implement proper forward (and backward) algorithm
class FeasibilityMeasure():
    def __init__(self, events, features, vocab_len):
        self.events = events
        self.features = features   
        self.vocab_len = vocab_len     
        self.tprobs = TransitionProbability(events, vocab_len)
        self.eprobs = EmissionProbabilityIndependentFeatures(events, features)

    def compute_valuation(self, events, features, is_joint=True, is_log=False):
        transition_probs = self.tprobs.compute_cum_probs(events, is_log)
        emission_probs = self.eprobs.compute_probs(events, features, is_log)
        results = transition_probs + emission_probs if is_log else transition_probs * emission_probs
 
        
        return results.sum(-1)[None] if is_log else results.prod(-1)[None]

    @property
    def transition_probabilities(self):
        return self.tprobs.transition_probs_matrix

    @property
    def emission_densities(self):
        return self.eprobs.gaussian_dists



if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    # generative_reader = GenerativeDataset(reader)
    (events, ev_features), _, _ = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=FeatureModes.FULL_SEP)
    metric = FeasibilityMeasure(events, ev_features)
    print(metric.compute_valuation(events, ev_features))

