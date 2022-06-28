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
from thesis_commons.distributions import ApproximateNormalDistribution, DataDistribution, GaussianParams, PFeaturesGivenActivity, TransitionProbability
from thesis_commons.random import matrix_sample, random

from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin
from thesis_viability.helper.base_distances import MeasureMixin

DEBUG = True
DEBUG_PRINT = True
if DEBUG_PRINT:
    np.set_printoptions(suppress=True)


# TODO: Implement proper forward (and backward) algorithm
class DatalikelihoodMeasure(MeasureMixin):
    def init(self, **kwargs) -> DatalikelihoodMeasure:
        if not self.data_distribution:
            raise Exception(f"Configuration is missing: data_distribution={self.data_distribution}")
        self.data_distribution = self.data_distribution.init()
        return self

    def set_data_distribution(self, data_distribution: DataDistribution) -> DatalikelihoodMeasure:
        self.data_distribution = data_distribution
        self.vocab_len = self.data_distribution.vocab_len
        self.max_len = self.data_distribution.max_len
        return self

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> DatalikelihoodMeasure:
        self.seq_lens = (cf_cases.events != 0).sum(axis=-1)[..., None]
        self.transition_probs, self.emission_probs = self.data_distribution.compute_probability(cf_cases) 
        self._results = (self.transition_probs * self.emission_probs)
        self._len_cases = len(fa_cases.events)
        return self

    @property
    def emission_densities(self):
        return self.data_distribution.eprobs.dists

    def _normalize_1(self, pure_results: np.ndarray):
        # Normalizing by number of actual steps
        indivisual_result = np.power(pure_results, 1 / np.maximum(self.seq_lens, 1))
        return indivisual_result

    def _normalize_2(self, pure_results: np.ndarray):
        # Normalizing each step seperately
        indivisual_result = np.power(pure_results, 1 / np.arange(1, pure_results.shape[1] + 1)[None])
        return indivisual_result

    def _normalize_3(self, pure_results: np.ndarray):
        # Normalizing by constant num of steps
        indivisual_result = np.power(pure_results, 1 / np.maximum(pure_results.shape[-1], 1))
        return indivisual_result

    def _normalize_4(self, pure_results: np.ndarray):
        # Omit Normalization
        indivisual_result = pure_results
        return indivisual_result

    def _aggregate_1(self, individuals):
        result = individuals.sum(-1, keepdims=True)
        return result

    def _aggregate_2(self, individuals):
        result = individuals.prod(-1, keepdims=True)
        return result

    def normalize_1(self, pure_results):
        individuals = self._normalize_2(pure_results)
        aggregated = self._aggregate_1(individuals)
        result = np.repeat(aggregated.T, self._len_cases, axis=0)
        normalized_results = result
        return normalized_results

    def normalize_2(self, pure_results):
        individuals = self._normalize_3(pure_results)
        aggregated = self._aggregate_2(individuals)
        result = np.repeat(aggregated.T, self._len_cases, axis=0)
        normalized_results = result
        return normalized_results

    def normalize(self):
        self.normalized_results = self.normalize_2(self._results)
        return self

    # https://stats.stackexchange.com/a/404643
    @property
    def result(self):
        results = self._aggregate_2(self._results)
        results = np.repeat(results.T, self._len_cases, axis=0)
        return results

    def sample(self, size=1):
        return self.data_distribution.sample(size)

    def get_config(self) -> BetterDict:
        return super().get_config().merge({"type": type(self).__name__})


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

        # NOTE: This makes no sense
        # class FeasibilityMeasureForwardIterative(DatalikelihoodMeasure):
        #     def compute_valuation(self, events, features, is_joint=True, is_log=False):
        #         #  https://github.com/katarinaelez/bioinformatics-algorithms/blob/master/hmm/hmm_guide.ipynb
        #         num_seq, seq_len, num_features = features.shape
        #         self.vocab_len
        #         events = events.astype(int)
        #         i = 0
        #         states = np.arange(self.vocab_len)
        #         all_possible_transitions = np.array(np.meshgrid(states, states)).reshape(2, -1).T
        #         from_state, to_state = all_possible_transitions[:, 0], all_possible_transitions[:, 1]
        #         all_possible_transitions[all_possible_transitions]
        #         trellis = np.zeros((num_seq, len(states), seq_len + 2))
        #         # emission_probs = self.eprobs.compute_probs(events, features)
        #         emission_probs = self.eprobs.compute_probs(events, features)
        #         flat_transitions = self.tprobs.extract_transitions(events)
        #         flat_transitions.reshape((num_seq, seq_len - 1, 2))
        #         transition_probs_matrix = self.tprobs.trans_probs_matrix
        #         self.tprobs.extract_transitions_probs(num_seq, flat_transitions)

        #         trellis[:, :, i + 1] = self.tprobs.start_probs[events[:, i]] * emission_probs[:, i, None]
        #         # Just to follow source example closely
        #         trellis[:, 0, 1] = 0
        #         trellis[:, :, 0] = 0
        #         trellis[:, 0, 0] = 1

        #         # for t in range(2, seq_len + 1):  # loops on the symbols
        #         #     for i in range(1, num_states - 1):  # loops on the states
        #         #         p_sum = trellis[:, :, t - 1].sum(-1) * transition_probs[events[:, i - 1], events[:, i]] * emission_probs[:, t - 1]

        #         # https://stats.stackexchange.com/a/31836
        #         # https://stats.stackexchange.com/a/254021
        #         for seq_idx in range(2, seq_len + 1):  # loops on the symbols
        #             emission_probs = emission_probs[:, seq_idx - 1][..., None]

        #             prev_vals = trellis[:, from_state, seq_idx - 1]  # all curr states with copies for each possible transition
        #             trans_probs = transition_probs_matrix[from_state, to_state][None]  # All transition combinations
        #             prev_vals * trans_probs

        #         results = None
        # return results.sum(-1)[None] if is_log else results.prod(-1)[None]
