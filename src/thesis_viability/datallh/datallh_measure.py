from __future__ import annotations
from collections import Counter, defaultdict
from ctypes import Union
from enum import IntEnum, auto
from typing import Any, Dict, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray
from numpy.linalg import LinAlgError
from scipy.stats._multivariate import \
    multivariate_normal_frozen as MultivariateNormal
from thesis_commons.distributions import ApproximateMultivariateNormal, DataDistribution, GaussianParams, PFeaturesGivenActivity, TransitionProbability
from thesis_commons.random import matrix_sample, random

from thesis_commons.representations import Cases
from thesis_viability.helper.base_distances import MeasureMixin

DEBUG = True
DEBUG_PRINT = True
if DEBUG_PRINT:
    np.set_printoptions(suppress=True)



# TODO: Call it data likelihood or possibility measure
# TODO: Implement proper forward (and backward) algorithm
class DatalikelihoodMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len, **kwargs):
        super(DatalikelihoodMeasure, self).__init__(vocab_len, max_len)

        training_data: Cases = kwargs.get('training_data', None)
        if training_data is None:
            raise ValueError("You need to provide training data for the Feasibility Measure")
        self.data_distribution = DataDistribution(training_data, vocab_len, max_len)

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> DatalikelihoodMeasure:
        seq_lens = (cf_cases.events != 0).sum(axis=-1)[..., None]
        # seq_lens = cf_cases.events.shape[-1]
        # seq_lens = np.arange(1, cf_cases.events.shape[-1]+1)
        transition_probs, emission_probs = self.data_distribution.pdf(cf_cases)
        # transition_probs, emission_probs = np.power(transition_probs, 1/seq_lens), np.power(emission_probs, 1/seq_lens)
        results = (transition_probs * emission_probs)
        results = np.power(results, 1 / seq_lens)
        results = results.prod(-1, keepdims=True)
        # results = np.power(results, 1/seq_lens)
        results_repeated = np.repeat(results.T, len(fa_cases.events), axis=0)
        self.results = results_repeated
        return self

    @property
    def transition_probabilities(self):
        return self.data_distribution.tprobs.trans_probs_matrix

    @property
    def emission_densities(self):
        return self.data_distribution.eprobs.gaussian_dists

    def normalize(self):
        self.normalized_results = self.results
        return self

    def sample(self, size=1):
        return self.data_distribution.sample(size)

# NOTE: This makes no sense ... Maybe it does...
class FeasibilityMeasureForward(DatalikelihoodMeasure):
    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases):
        T = fa_cases.events.shape[-1] - 1
        results = self.forward_algorithm(fa_cases.events.astype(int), fa_cases.features, T)

        self.results = results

    def forward_algorithm(self, events, features, t):
        if t == 0:
            xt = events[:, t]
            return self.initial_trans_probs[xt]

        xt, xt_prev = events[:, t, None], events[:, t - 1, None]
        yt = features[:, t]
        p_yt_given_xt = self.eprobs(yt, xt)
        recursion_part = np.array([self.tprobs(xt, xt_prev) * self.forward_algorithm(events, features, t_sub) for t_sub in range(t)])
        at_xt = p_yt_given_xt * np.sum(recursion_part, axis=0)

        return at_xt

    # unique_events = np.unique(xt)
    # emission_probs = np.zeros_like(xt)
    # for ev in unique_events:
    #     print(ev)
    #     ev_pos = xt == ev
    #     print(ev_pos.T)
    #     distribution = self.eprobs.gaussian_dists.get(ev, None)
    #     print(distribution.mean)
    #     print(yt[ev_pos.flatten()])
    #     emission_probs[ev_pos] = distribution.pdf(yt[ev_pos.flatten()]) if distribution else 0
    # print(emission_probs.T)

    # ev = 1
    # print(ev)
    # ev_pos = xt == ev
    # print(ev_pos.T)
    # distribution = self.eprobs.gaussian_dists.get(ev, None)
    # print(distribution.mean)
    # print(yt[ev_pos.flatten()])
    # print(distribution.pdf(yt[ev_pos.flatten()]))


# NOTE: This makes no sense
class FeasibilityMeasureForwardIterative(DatalikelihoodMeasure):
    def compute_valuation(self, events, features, is_joint=True, is_log=False):
        #  https://github.com/katarinaelez/bioinformatics-algorithms/blob/master/hmm/hmm_guide.ipynb
        num_seq, seq_len, num_features = features.shape
        self.vocab_len
        events = events.astype(int)
        i = 0
        states = np.arange(self.vocab_len)
        all_possible_transitions = np.array(np.meshgrid(states, states)).reshape(2, -1).T
        from_state, to_state = all_possible_transitions[:, 0], all_possible_transitions[:, 1]
        all_possible_transitions[all_possible_transitions]
        trellis = np.zeros((num_seq, len(states), seq_len + 2))
        # emission_probs = self.eprobs.compute_probs(events, features)
        emission_probs = self.eprobs.compute_probs(events, features)
        flat_transitions = self.tprobs.extract_transitions(events)
        flat_transitions.reshape((num_seq, seq_len - 1, 2))
        transition_probs_matrix = self.tprobs.trans_probs_matrix
        self.tprobs.extract_transitions_probs(num_seq, flat_transitions)

        trellis[:, :, i + 1] = self.tprobs.start_probs[events[:, i]] * emission_probs[:, i, None]
        # Just to follow source example closely
        trellis[:, 0, 1] = 0
        trellis[:, :, 0] = 0
        trellis[:, 0, 0] = 1

        # for t in range(2, seq_len + 1):  # loops on the symbols
        #     for i in range(1, num_states - 1):  # loops on the states
        #         p_sum = trellis[:, :, t - 1].sum(-1) * transition_probs[events[:, i - 1], events[:, i]] * emission_probs[:, t - 1]

        # https://stats.stackexchange.com/a/31836
        # https://stats.stackexchange.com/a/254021
        for seq_idx in range(2, seq_len + 1):  # loops on the symbols
            emission_probs = emission_probs[:, seq_idx - 1][..., None]

            prev_vals = trellis[:, from_state, seq_idx - 1]  # all curr states with copies for each possible transition
            trans_probs = transition_probs_matrix[from_state, to_state][None]  # All transition combinations
            prev_vals * trans_probs

        results = None
        return results.sum(-1)[None] if is_log else results.prod(-1)[None]
