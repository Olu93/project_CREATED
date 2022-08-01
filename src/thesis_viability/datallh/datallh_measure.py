from __future__ import annotations
from collections import Counter, defaultdict
from ctypes import Union
from enum import IntEnum, auto
from typing import Any, Dict, Sequence, Tuple, TypedDict

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.stats as stats
from numpy.linalg import LinAlgError
from scipy.stats._multivariate import \
    multivariate_normal_frozen as MultivariateNormal
from thesis_commons.distributions import DataDistribution, GaussianParams, PFeaturesGivenActivity, TransitionProbability
from thesis_commons.random import matrix_sample, random

from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin
from thesis_viability.helper.base_distances import MeasureMixin
from scipy import special

DEBUG = True
DEBUG_PRINT = True
DEBUG_NORMALIZE = 0
if DEBUG_PRINT:
    np.set_printoptions(suppress=True)


# TODO: Implement proper forward (and backward) algorithm
# TODO: Explore ways to prevent underflow -- "process mining" AND underflow AND perplexity
# -- https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
# -- https://surge-ai.medium.com/evaluating-language-models-an-introduction-to-perplexity-in-nlp-f6019f7fb914
# -- https://www.annytab.com/dynamic-bayesian-network-in-python/
class DatalikelihoodMeasure(MeasureMixin):
    def init(self, **kwargs) -> DatalikelihoodMeasure:
        if not self.data_distribution:
            raise Exception(f"Configuration is missing: data_distribution={self.data_distribution}")
        self.data_distribution = self.data_distribution.init()
        self.norm_method = kwargs.pop('norm',0) 
        return self

    def set_data_distribution(self, data_distribution: DataDistribution) -> DatalikelihoodMeasure:
        self.data_distribution = data_distribution
        self.vocab_len = self.data_distribution.vocab_len
        self.max_len = self.data_distribution.max_len
        return self

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> DatalikelihoodMeasure:
        self._cf_seq_lens_mask, self._cf_seq_lens, self._cf_len = self.extract_helpers(cf_cases)
        self._fa_seq_lens_mask, self._fa_seq_lens, self._fa_len = self.extract_helpers(fa_cases)
        self.transition_probs, self.emission_probs = self.data_distribution.compute_probability(cf_cases)
        self._results = (self.transition_probs * self.emission_probs)
        self.fa_events = fa_cases.events
        self.cf_events = cf_cases.events
        return self

    def extract_helpers(self, fa_cases: Cases):
        _fa_seq_lens_mask = ~((fa_cases.events != 0).cumsum(-1) > 0)
        _fa_seq_lens = (~_fa_seq_lens_mask).sum(axis=-1, keepdims=True)
        _fa_len = len(fa_cases.events)
        return _fa_seq_lens_mask, _fa_seq_lens, _fa_len

    @property
    def emission_densities(self):
        return self.data_distribution.eprobs.dists

    def normalize_1(self, x: np.ndarray) -> np.ndarray:
        # Omit Normalization
        x = np.log(x + np.finfo(float).eps).sum(-1, keepdims=True)
        x = np.exp(x)
        x = np.repeat(x.T, self._fa_len, axis=0)
        return x

    def normalize_2(self, x: np.ndarray) -> np.ndarray:
        # Normalize: By number of real steps in cf
        x = np.log(x + np.finfo(float).eps).sum(-1, keepdims=True)
        x = np.exp(x / np.maximum(self._cf_seq_lens, 1))
        x = np.repeat(x.T, self._fa_len, axis=0)
        return x

    def normalize_3(self, x: np.ndarray) -> np.ndarray:
        # Normalize: Rewarding sequence by how close to the orginal sequence length
        x = np.log(x + np.finfo(float).eps).sum(-1, keepdims=True)
        x = np.exp(x * (np.maximum(np.abs(self._cf_seq_lens - self._fa_seq_lens), 1) / self.max_len))
        x = np.repeat(x.T, self._fa_len, axis=0)
        return x

    def normalize_4(self, x: np.ndarray) -> np.ndarray:
        # Normalize: Rewarding the longest sequence with the highest average feasibility
        x = np.log(x + np.finfo(float).eps).sum(-1, keepdims=True)
        x = np.exp(x * (self._cf_seq_lens / self.max_len))
        x = np.repeat(x.T, self._fa_len, axis=0)
        return x

    # https://mmeredith.net/blog/2017/UnderOverflow.htm
    # https://stackoverflow.com/a/26436494/4162265
    # https://stats.stackexchange.com/questions/464096/summation-of-log-probabilities
    # Normalization https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    # https://stackoverflow.com/a/52132033/4162265
    def normalize_5(self, x: np.ndarray) -> np.ndarray:
        x = np.exp(1 - special.logsumexp(x, axis=-1, keepdims=True))
        x = np.repeat(x.T, self._fa_len, axis=0)
        return x


    def normalize(self):
        tmp_results = self._results
        # Masking case:
        if DEBUG_NORMALIZE:
            tmp = self.check_nom_results(tmp_results)


        tmp_results = ma.masked_array(tmp_results, self._cf_seq_lens_mask)
        tmp = self.normalize_2(tmp_results)

        self.normalized_results = tmp
        return self

    def check_nom_results(self, tmp_results):
        print("============= MASKED =============")
        tmp_results = ma.masked_array(tmp_results, self._cf_seq_lens_mask)
        tmp = self.normalize_1(tmp_results)
        self.debug_results("tmp1", tmp.data)  # Optimizes shortness
        tmp = self.normalize_2(tmp_results)
        self.debug_results("tmp2", tmp.data) # Best without the use of fa_case
        # tmp = self.normalize_3(tmp_results)
        # self.debug_results("tmp3", tmp.data) # Good but uses fa_case
        # tmp = self.normalize_4(tmp_results)
        # self.debug_results("tmp4", tmp.data) # 1 and 4 are identical and optimze shortness

        print("============= NOT-MASKED =============")
        tmp_results = self._results
        # tmp = self.normalize_1(tmp_results) 
        # self.debug_results("tmp1", tmp)  # 1 and 4 are identical and optimze shortness
        # tmp = self.normalize_2(tmp_results)
        # self.debug_results("tmp2", tmp) # No difference to with mask
        # tmp = self.normalize_3(tmp_results)
        # self.debug_results("tmp3", tmp) # Good but uses fa_case
        # tmp = self.normalize_4(tmp_results)
        # self.debug_results("tmp4", tmp) # 1 and 4 are identical and optimze shortness
        tmp = self.normalize_5(tmp_results)
        self.debug_results("tmp5", tmp) # Returns good manual tasks 
        print("============= DEBUG-END =============")
        return tmp

    def debug_results(self, lbl, x):
        print(f"---------------- {lbl} start ----------------")
        print(self.fa_events[0])
        print(x[0].argmax())
        print(x[0][x[0].argmax()-2: x[0].argmax()+3][..., None])
        print(self.cf_events[x[0].argmax()-2: x[0].argmax()+3])
        print(f"---------------- {lbl}: end ----------------")

    # https://stats.stackexchange.com/a/404643

    @property
    def result(self) -> np.ndarray:
        results = self._aggregate_2(self._results)
        results = np.repeat(results.T, self._fa_len, axis=0)
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
