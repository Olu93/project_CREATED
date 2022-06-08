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

from thesis_commons.representations import Cases
from thesis_viability.helper.base_distances import MeasureMixin

DEBUG = True
DEBUG_PRINT = True
if DEBUG_PRINT:
    np.set_printoptions(precision=5, suppress=True)


# https://gist.github.com/righthandabacus/f1d71945a49e2b30b0915abbee668513
def sliding_window(events, win_size) -> NDArray:
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
        self.trans_count_matrix: NDArray = np.zeros((vocab_len, vocab_len))
        self.trans_probs_matrix: NDArray = np.zeros((vocab_len, vocab_len))

        self.df_tra_counts = pd.DataFrame(events_slided.reshape((-1, 2)).tolist()).value_counts()
        self.trans_idxs = np.array(self.df_tra_counts.index.tolist(), dtype=int)
        self.trans_from = self.trans_idxs[:, 0]
        self.trans_to = self.trans_idxs[:, 1]
        self.trans_counts = np.array(self.df_tra_counts.values.tolist(), dtype=int)
        self.trans_count_matrix[self.trans_from, self.trans_to] = self.trans_counts
        self.trans_probs_matrix = self.trans_count_matrix / self.trans_count_matrix.sum(axis=1, keepdims=True)
        self.trans_probs_matrix[np.isnan(self.trans_probs_matrix)] = 0

        self.start_count_matrix = np.zeros((vocab_len, 1))
        self.start_events = self.events[:, 0]
        self.start_counts_counter = Counter(self.start_events)
        self.start_indices = np.array(list(self.start_counts_counter.keys()), dtype=int)
        self.start_counts = np.array(list(self.start_counts_counter.values()), dtype=int)
        self.start_count_matrix[self.start_indices, 0] = self.start_counts
        self.start_probs = self.start_count_matrix / self.start_counts.sum()

    def compute_sequence_probabilities(self, events, is_joint=True) -> NDArray:
        flat_transistions = self.extract_transitions(events)
        probs = self.extract_transitions_probs(events.shape[0], flat_transistions)
        start_events = np.array(list(events[:, 0]), dtype=int)
        start_event_prob = self.start_probs[start_events, 0, None]
        result = np.hstack([start_event_prob, probs])
        return result.prod(-1) if is_joint else result

    def extract_transitions_probs(self, num_events, flat_transistions) -> NDArray:
        t_from = flat_transistions[:, 0]
        t_to = flat_transistions[:, 1]
        probs = self.trans_probs_matrix[t_from, t_to].reshape(num_events, -1)
        return probs

    def extract_transitions(self, events) -> NDArray:
        events_slided = sliding_window(events, 2)
        events_slided_flat = events_slided.reshape((-1, 2))
        transistions = np.array(events_slided_flat.tolist(), dtype=int)
        return transistions

    def compute_sequence_logprobabilities(self, events, is_joint=True) -> NDArray:
        sequential_probabilities = self.compute_sequence_probabilities(events, False)
        log_probs = np.log(sequential_probabilities + np.finfo(float).eps)
        stable_joint_log_probs = log_probs.sum(-1)
        return stable_joint_log_probs if is_joint else log_probs

    def compute_cum_probs(self, events, is_log=False) -> NDArray:
        sequence_probs = self.compute_sequence_probabilities(events, False).cumprod(-1) if not is_log else self.compute_sequence_logprobabilities(events, False).cumsum(-1)
        return sequence_probs

    def __call__(self, xt: NDArray, xt_prev: NDArray) -> NDArray:
        probs = self.trans_probs_matrix[xt_prev, xt]
        return probs


class GaussianParams():
    mean: NDArray = None
    cov: NDArray = None
    support: int = 0

    def __init__(self, mean: NDArray, cov: NDArray, support: NDArray, key: Any = None):
        self.key = key
        self.mean = mean
        self.cov = cov if support != 1 else np.zeros_like(cov)
        self.support = support
        self._dim = self.mean.shape[0]

    def set_cov(self, cov: NDArray) -> GaussianParams:
        self.cov = cov
        return self

    def set_mean(self, mean: NDArray) -> GaussianParams:
        self.mean = mean
        return self

    def set_support(self, support: int) -> GaussianParams:
        self.support = support
        return self

    def __repr__(self) -> str:
        return f"@GaussianParams[len={self.support}]"

    @property
    def dim(self) -> int:
        return self._dim


class ApproximationLevel():
    # DEFAULT = auto()
    # FILL = auto()
    # EPS = auto()
    # SAMPLE = auto()
    # NONE = auto()
    # LAST_RESORT = auto()
    # ALL_FAILED = -1
    def __init__(self, is_eps: int, approx_type: int):
        self.eps_type = is_eps
        self.approx_type = approx_type
        self._mapping_eps = {
            0: "EPS_FALSE",
            1: "EPS_TRUE",
            2: "NONE",
        }
        self._mapping_apprx = {
            0: "DEFAULT",
            1: "FILL",
            2: "SAMPLE",
            3: "ONLY_MEAN",
            4: "LAST_RESORT",
        }
        self._justification_eps = max([len(s) for s in self._mapping_eps.values()])
        self._justification_apprx = max([len(s) for s in self._mapping_apprx.values()])

    def __repr__(self):
        eps_string = self._mapping_eps.get(self.eps_type, "UNDEFINED")
        approx_string = self._mapping_apprx.get(self.approx_type, "UNDEFINED")
        # https://www.geeksforgeeks.org/pad-or-fill-a-string-by-a-variable-in-python-using-f-string/
        return f"{self.eps_type}{self.approx_type}_{eps_string:_<{self._justification_eps}}_{approx_string:<{self._justification_apprx}}"


class ApproximateMultivariateNormal():
    class FallbackableException(Exception):
        def __init__(self, e: Exception) -> None:
            super().__init__(*e.args)

    def __init__(self, params: GaussianParams, fallback_params: GaussianParams = None, eps: float = None):
        self.event = params.key
        self.params = params
        self.fallback_params = fallback_params
        self.eps = eps
        self.dist, self.approximation_level = self.init_dists(self.params, self.fallback_params, self.eps)

    def init_dists(self, params: GaussianParams, fallback_params: GaussianParams, eps: float) -> Tuple[MultivariateNormal, ApproximationLevel]:
        dist = None

        no_eps = np.zeros_like(params.cov)
        diagonal_eps = np.identity(params.dim) * eps
        everywhere_eps = np.ones_like(params.cov) * eps
        str_event = f"Event {self.event:02d}"
        for i, eps in enumerate([no_eps, diagonal_eps, everywhere_eps]):
            cov_unchanged = params.cov + eps
            cov_filled = np.where(params.cov == 0, fallback_params.cov, params.cov) + eps
            for j, cov in enumerate([cov_unchanged, cov_filled]):
                dist = self._create_multivariate(params.mean, cov)
                state, is_error, str_result = self._process_dist_result(dist, str_event, i, j)
                print(str_result)
                if not is_error:
                    return dist, state

        for i, eps in enumerate([no_eps, diagonal_eps, everywhere_eps]):
            dist = self._create_multivariate(params.mean, fallback_params.cov + eps)
            j = 3
            state, is_error, str_result = self._process_dist_result(dist, str_event, i, j)
            print(str_result)
            if not is_error:
                return dist, state

        state = ApproximationLevel(2, 4)
        print(f"WARNING {str_event}: Could not any approx for event {self.event} -- {state}")
        return stats.multivariate_normal(np.zeros_like(params.mean)), state

    def _process_dist_result(self, dist, str_event, i, j):
        state = ApproximationLevel(i, j)
        is_error = (type(dist) == ApproximateMultivariateNormal.FallbackableException)
        str_result = f"WARNING {str_event}: {state} -- Could not create because {dist}" if is_error else f"SUCCESS {str_event}: {state}"
        return state, is_error, str_result

    def _create_multivariate(self, mean: NDArray, cov: NDArray) -> Union[FallbackableException, MultivariateNormal]:
        try:
            return stats.multivariate_normal(mean, cov)
        except LinAlgError as e:
            return ApproximateMultivariateNormal.FallbackableException(e)
        except ValueError as e:
            if e.args[0] == "the input matrix must be positive semidefinite":
                return ApproximateMultivariateNormal.FallbackableException(e)
            else:
                raise e

    def pdf(self, x: NDArray):
        return self.dist.pdf(x)

    @property
    def mean(self):
        return self.dist.mean

    @property
    def cov(self):
        return self.dist.cov

    def __repr__(self):
        return f"@{type(self).__name__}[Fallback {self.approximation_level} - Mean: {self.mean}]"


class NoneExistingMultivariateNormal(ApproximateMultivariateNormal):
    def __init__(self, params: GaussianParams = None, fallback_params: GaussianParams = None, eps: float = None):
        self.approximation_level = ApproximationLevel(-1, -1)

    def pdf(self, x: NDArray):
        return np.zeros((len(x),))

    @property
    def mean(self):
        return None

    @property
    def cov(self):
        return None


class PFeaturesGivenActivity():
    def __init__(self, all_dists: Dict[int, ApproximateMultivariateNormal]):
        self.all_dists = all_dists
        self.none_existing_dists = NoneExistingMultivariateNormal()

    def __getitem__(self, key) -> ApproximateMultivariateNormal:
        if key in self.all_dists:
            return self.all_dists.get(key)
        else:
            return self.none_existing_dists

    def __repr__(self):
        return f"@{type(self).__name__}{self.all_dists}"


class EmissionProbability():

    # TODO: Create class that simplifies the dists to assume feature independence
    def __init__(self, events: NDArray, features: NDArray, eps: float = 0.1):
        num_seq, seq_len, num_features = features.shape
        self.events = events
        self.features = features
        events_flat = self.events.reshape((-1, ))
        features_flat = self.features.reshape((-1, num_features))
        sort_indices = events_flat.argsort()
        events_sorted = events_flat[sort_indices]
        features_sorted = features_flat[sort_indices]
        self.eps = eps
        self.df_ev_and_ft: pd.DataFrame = pd.DataFrame(features_sorted)
        self.data_groups: Dict[int, pd.DataFrame] = {}
        self.df_ev_and_ft["event"] = events_sorted.astype(int)

        self.data_groups, self.gaussian_params, self.gaussian_dists = self.estimate_params()

    def compute_probs(self, events: NDArray, features: NDArray, is_log=False) -> NDArray:
        num_seq, seq_len, num_features = features.shape
        events_flat = events.reshape((-1, )).astype(int)
        features_flat = features.reshape((-1, num_features))
        unique_events = np.unique(events_flat)
        emission_probs = np.zeros_like(events_flat, dtype=float)
        for ev in unique_events:
            # https://stats.stackexchange.com/a/331324
            ev_pos = events_flat == ev
            if ev == 37:
                print("STOP")
            distribution = self.gaussian_dists[ev]
            emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])
            # distribution = self.gaussian_dists[ev]
            # emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])

        result = emission_probs.reshape((num_seq, -1))
        return np.log(result) if is_log else result

    def estimate_params(self) -> Tuple[Dict[int, pd.DataFrame], Dict[int, GaussianParams], PFeaturesGivenActivity]:
        data = self.df_ev_and_ft.drop('event', axis=1)
        fallback = self.extract_fallback_params(data)
        data_groups = self.extract_data_groups(self.df_ev_and_ft)
        gaussian_params = self.extract_gaussian_params(data_groups)
        gaussian_dists = self.extract_gaussian_dists(gaussian_params, fallback, self.eps)

        return data_groups, gaussian_params, gaussian_dists

    def extract_fallback_params(self, data):
        fallback = GaussianParams(data.mean().values, data.cov().values, len(data))
        return fallback

    def extract_data_groups(self, dataset: pd.DataFrame) -> Dict[int, ApproximateMultivariateNormal]:
        return {key: data.loc[:, data.columns != 'event'] for (key, data) in dataset.groupby("event")}

    def extract_gaussian_params(self, data_groups: Dict[int, pd.DataFrame]) -> Dict[int, GaussianParams]:
        return {activity: GaussianParams(data.mean().values, data.cov().values, len(data), activity) for activity, data in data_groups.items()}

    def extract_gaussian_dists(self, gaussian_params: Dict[int, GaussianParams], fallback: GaussianParams, eps: float):
        return PFeaturesGivenActivity({activity: ApproximateMultivariateNormal(data, fallback, eps) for activity, data in gaussian_params.items()})


class EmissionProbabilityIndependentFeatures(EmissionProbability):
    def extract_fallback_params(self, data) -> GaussianParams:
        params = super().extract_fallback_params(data)
        return params.set_cov(params.cov * np.eye(*params.cov.shape))

    def extract_gaussian_params(self, data_groups: Dict[int, pd.DataFrame]) -> Dict[int, GaussianParams]:
        return {key: data.set_cov(data.cov * np.eye(*data.cov.shape)) for key, data in super().extract_gaussian_params(data_groups).items()}


# TODO: Call it data likelihood or possibility measure
# TODO: Implement proper forward (and backward) algorithm
class DatalikelihoodMeasure(MeasureMixin):
    def __init__(self, vocab_len, max_len, **kwargs):
        super(DatalikelihoodMeasure, self).__init__(vocab_len, max_len)

        training_data: Cases = kwargs.get('training_data', None)
        if training_data is None:
            raise ValueError("You need to provide training data for the Feasibility Measure")
        events, features = training_data.cases
        self.events = events
        self.features = features
        self.vocab_len = vocab_len
        self.tprobs = TransitionProbability(events, vocab_len)
        self.eprobs = EmissionProbabilityIndependentFeatures(events, features)
        self.transition_probs = self.tprobs.trans_probs_matrix
        self.emission_dists = self.eprobs.gaussian_dists
        self.initial_trans_probs = self.tprobs.start_probs

    def compute_valuation(self, fa_cases: Cases, cf_cases: Cases) -> DatalikelihoodMeasure:
        transition_probs = self.tprobs.compute_cum_probs(cf_cases.events, is_log=False)
        emission_probs = self.eprobs.compute_probs(cf_cases.events, cf_cases.features, is_log=False)
        results = (transition_probs * emission_probs).prod(-1)[None]
        seq_lens = (cf_cases.events != 0).sum(axis=-1)[None]
        # results = np.power(results, 1/seq_lens) #TODO AAAAAAAAAAAAAAAAAAA Normalization according to length
        results_repeated = np.repeat(results, len(fa_cases.events), axis=0)
        self.results = results_repeated
        return self

    @property
    def transition_probabilities(self):
        return self.tprobs.transition_probs_matrix

    @property
    def emission_densities(self):
        return self.eprobs.gaussian_dists

    # def normalize(self):
    #     normed_values = self.results / self.results.sum(axis=1, keepdims=True)
    #     self.normalized_results = normed_values
    #     return self

    def normalize(self):
        self.normalized_results = self.results
        return self


# NOTE: This makes no sense
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
