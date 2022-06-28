from __future__ import annotations
from abc import ABC, abstractmethod
import itertools as it
from typing import TYPE_CHECKING, Callable, List, Tuple, Any, Dict, Sequence, Tuple, TypedDict
import sys
from thesis_commons.config import DEBUG_DISTRIBUTION, FIX_BINARY_OFFSET
from thesis_commons.functions import sliding_window
from thesis_commons.random import matrix_sample
from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, ConfigurationSet
if TYPE_CHECKING:
    pass

from collections import Counter, defaultdict
from ctypes import Union
from enum import IntEnum, auto

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.linalg import LinAlgError
from scipy.stats._distn_infrastructure import rv_frozen as ScipyDistribution
# from scipy.stats._discrete_distns import rv_discrete
from pandas.core.groupby.generic import DataFrameGroupBy

EPS = np.finfo(float).eps


def is_invertible(a):  # https://stackoverflow.com/a/17931970/4162265
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def is_singular(a):  # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    return np.linalg.cond(a) < 1 / sys.float_info.epsilon


class ResultTransitionProb():
    def __init__(self, seq_probs: np.ndarray):
        self.pnt_p = seq_probs
        self.pnt_log_p = np.log(self.pnt_p + EPS)
        # self.seq_log_p = self.pnt_log_p.cumsum(-1)
        # self.seq_p = np.exp(self.pnt_log_p)
        # self.jnt_pnt_p = self.pnt_p.prod(-1)
        # self.jnt_pnt_log_p = self.pnt_log_p.sum(-1)
        # self.jnt_seq_p = self.seq_p.prod(-1)
        # self.jnt_seq_log_p = self.seq_log_p.sum(-1)


class ProbabilityMixin:
    def set_vocab_len(self, vocab_len: int) -> ProbabilityMixin:
        self.vocab_len = vocab_len
        return self

    def set_max_len(self, max_len: int) -> ProbabilityMixin:
        self.max_len = max_len
        return self

    def set_data(self, cases: Cases) -> ProbabilityMixin:
        self.events = cases.events
        self.features = cases.features
        return self

    def set_data_mapping(self, data_mapping: Dict) -> ProbabilityMixin:
        self.data_mapping = data_mapping
        return self


class TransitionProbability(ProbabilityMixin, ABC):
    def compute_probs(self, events, logdomain=False, joint=False, cummulative=True, **kwargs) -> np.ndarray:
        res = ResultTransitionProb(self._compute_p_seq(events, **kwargs))
        result = res.pnt_log_p
        if not (cummulative or joint or logdomain):
            return np.exp(result)
        result = np.cumsum(result, -1) if cummulative else result
        result = np.sum(result, -1) if joint else result
        result = result if logdomain else np.exp(result)
        return result

    @abstractmethod
    def init(self, events, **kwargs):
        pass

    @abstractmethod
    def _compute_p_seq(self, events: np.ndarray) -> np.ndarray:
        return None

    @abstractmethod
    def extract_transitions_probs(self, num_events: int, flat_transistions: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def extract_transitions(self, events: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, sample_size: int) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, xt: np.ndarray, xt_prev: np.ndarray) -> np.ndarray:
        pass


class DistParams(ABC):
    def __init__(self):
        self.data = None
        self.support = None
        self.key = None
        self.idx_features = None
        self.dist: ScipyDistribution = None

    @abstractmethod
    def init(self) -> DistParams:
        return self

    def set_data(self, data: pd.DataFrame) -> DistParams:
        self.data = data
        return self

    def set_support(self, support: int) -> DistParams:
        self.support = support
        return self

    def set_key(self, key: str) -> DistParams:
        self.key = key
        return self

    def set_idx_features(self, idx_features: List[int]) -> DistParams:
        self.idx_features = idx_features
        self.feature_len = len(self.idx_features)
        return self

    @abstractmethod
    def sample(self, size) -> np.ndarray:
        pass

    @abstractmethod
    def compute_probability(self, data: np.ndarray):
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"@{type(self).__name__}[len={self.support}]"


class DiscreteDistribution(DistParams):
    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        if ((data.max() != 0) or (data.min() != 0)) and DEBUG_DISTRIBUTION:
            print("STOP")
        tmp = (data > 0) * 1
        result = self.dist.pmf(tmp)
        if np.any(np.isnan(result)) and DEBUG_DISTRIBUTION:
            print("STOP")
        if (len(result.shape) > 2) and DEBUG_DISTRIBUTION:
            print("Stop")
        return result

    def sample(self, size) -> np.ndarray:
        return self.dist.rvs(size)


class ContinuousDistribution(DistParams):
    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        if (len(data) < 2) and DEBUG_DISTRIBUTION:
            print("Stop")
        result = self.dist.pdf(data[:, self.idx_features]) if len(data) > 1 else np.array([self.dist.pdf(data[:, self.idx_features])])
        if np.any(np.isnan(result)) and DEBUG_DISTRIBUTION:
            print("STOP")
        return result

    def sample(self, size) -> np.ndarray:
        return self.dist.rvs(size)


class BernoulliParams(DiscreteDistribution):
    def init(self) -> BernoulliParams:
        self._data: np.ndarray = self.data[self.idx_features]
        self._counts = self._data.sum()
        self._p = self._counts / len(self._data)
        self.dist = stats.bernoulli(p=self._p)
        return self

    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        result = super().compute_probability(data[:, self.idx_features])
        if (len(result.shape) > 2) and DEBUG_DISTRIBUTION:
            print("Stop")
        return result

    def sample(self, size) -> np.ndarray:
        sampled = (np.random.uniform(size=(size, self.feature_len)) < self._p.values[None])
        result = sampled * 1
        return result

    @property
    def p(self) -> float:
        return self._p

    def __repr__(self):
        string = super().__repr__()
        return f"{string} -- {self._counts}"


# https://distribution-explorer.github.io/discrete/categorical.html
class MultinoulliParams(DiscreteDistribution):
    def init(self) -> MultinoulliParams:
        self._data: pd.DataFrame = self.data[self.idx_features]
        self._positions = np.where(self._data.values == 1)
        self._cols = self._positions[1]
        self._unique_vals, self._counts = np.unique(self._cols, return_counts=True)

        self._p = self._counts / self._counts.sum()
        if not self._p.size == 0:
            self.dist = stats.rv_discrete(values=(self._unique_vals, self._p))
        return self

    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        positions = np.where(data[:, self.idx_features] == 1)
        cols = positions[1]
        if cols.size == 0:
            return np.ones((len(data), 1))
        result = self.dist.pmf(cols[:, None])
        return result

    @property
    def p(self) -> float:
        return self._p

    def __repr__(self):
        string = super().__repr__()
        return f"{string} -- {self._counts}"


# https://www.universalclass.com/articles/math/statistics/calculate-probabilities-normally-distributed-data.htm
# https://www.youtube.com/watch?v=Dn6b9fCIUpM
class GaussianParams(ContinuousDistribution):
    eps = EPS + 0.00001

    def init(self) -> GaussianParams:
        self._data: np.ndarray = self.data[self.idx_features]
        self._mean: np.ndarray = self._data.mean().values
        self._cov: np.ndarray = self._data.cov().values if self.support != 1 else np.zeros_like(self._data.cov())
        self._var: np.ndarray = self._data.var().values
        # self.key = self.key
        # self.cov_mask = self.compute_cov_mask(self._cov)
        self.dist = self.create_dist()

        return self

    def create_dist(self):
        if (self._mean.sum() == 0) and (self._cov.sum() == 0):
            print(f"Activity {self.key}: Could not create Gaussian for -- Mean and Covariance are zero -> Use default")
            dist = stats.multivariate_normal(self._mean, self._cov, allow_singular=True)
            return dist

        try:
            dist = stats.multivariate_normal(self._mean, self._cov)
            print(f"Activity {self.key}: Use proper distribution")
            return dist
        except Exception as e:
            print(f"Activity {self.key}: Could not create Gaussian for -- {e} -> use col imputation")

        try:
            # check_if_cov_unique = (self._cov - self._cov.mean(axis=1)[None])**2
            # rank = np.linalg.matrix_rank(self._cov)
            is_singular_col = self._var == 0

            # maximum = np.maximum(, EPS)
            tmp = self._cov.copy()
            tmp_diag = np.diag(tmp).copy()
            tmp_diag[is_singular_col] = tmp_diag[is_singular_col] + EPS
            new_cov = np.diag(tmp_diag)
            dist = stats.multivariate_normal(self._mean, new_cov)
            return dist
        except Exception as e:
            print(f"Activity {self.key}: Could not create Gaussian for -- {e} -> use imputed constants...")

        try:

            tmp = self._cov.copy() + EPS
            tmp_diag = np.diag(tmp).copy()
            new_cov = np.diag(tmp_diag)
            dist = stats.multivariate_normal(self._mean, new_cov)
            return dist
        except Exception as e:
            print(f"Activity {self.key}: Could not create Gaussian for -- {e} -> use adding eps to all vals...")

        try:
            new_cov = np.diag(self._var + EPS)
            dist = stats.multivariate_normal(self._mean, new_cov)
            return dist
        except Exception as e:
            print(f"Activity {self.key}: Could not create Gaussian for -- {e} -> use Variance only")

        # try:
        #     new_cov = np.eye(*self._cov.shape) * EPS
        #     dist = stats.multivariate_normal(self._mean, new_cov)
        #     return dist
        # except Exception as e:
        #     print(f"Standard Normal Dist: Could not create Gaussian for Activity {self.key}: {e}")

        print(f"Activity {self.key}: Could not create Gaussian for -- {'Everything failed!'} - use malformed...")
        dist = stats.multivariate_normal(self._mean, self._cov, allow_singular=True)
        return dist

    # def compute_cov_mask(self, cov: np.ndarray) -> np.ndarray:
    #     row_sums = cov.sum(0)[None, ...] == 0
    #     col_sums = cov.sum(1)[None, ...] == 0
    #     masking_pos = ~((row_sums).T | (col_sums))
    #     return masking_pos

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        result = super().compute_probability(data)[:, None]
        return result


class IndependentGaussianParams(GaussianParams):
    def init(self) -> IndependentGaussianParams:
        super().init()
        self._cov: np.ndarray = np.diag(np.diag(self._cov)) if self.support > 1 else np.zeros_like(self._cov)
        return self


class ApproximationLevel():
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
            5: "ALLOW_DEGNRT",
        }
        self._justification_eps = max([len(s) for s in self._mapping_eps.values()])
        self._justification_apprx = max([len(s) for s in self._mapping_apprx.values()])

    def __repr__(self):
        eps_string = self._mapping_eps.get(self.eps_type, "UNDEFINED")
        approx_string = self._mapping_apprx.get(self.approx_type, "UNDEFINED")
        # https://www.geeksforgeeks.org/pad-or-fill-a-string-by-a-variable-in-python-using-f-string/
        return f"{self.eps_type}{self.approx_type}_{eps_string:_<{self._justification_eps}}_{approx_string:<{self._justification_apprx}}"


class FallbackableException(Exception):
    def __init__(self, e: Exception) -> None:
        super().__init__(*e.args)


class Dist:
    def __init__(self, key):
        self.event = key

    def set_event(self, event: int) -> Dist:
        self.event = event
        return self

    def set_params(self, params: Dict) -> Dist:
        self.params = params
        return self

    def set_support(self, support: int) -> Dist:
        self.support = support
        return self

    def set_fallback_params(self, fallback_params: Dict) -> Dist:
        self.fallback_params = fallback_params
        return self

    def set_feature_len(self, feature_len: int) -> Dist:
        self.feature_len = feature_len
        return self

    @abstractmethod
    def init(self, **kwargs) -> Dist:
        pass

    @abstractmethod
    def rvs(self, size: int) -> np.ndarray:
        pass

    @abstractmethod
    def compute_joint_p(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self):
        return f"@{type(self).__name__}"


class MixedDistribution(Dist):
    def __init__(self, key):
        super().__init__(key)
        self.distributions: List[DistParams] = []

    def init(self, **kwargs) -> MixedDistribution:
        for dist in self.distributions:
            dist.init()
        return self

    def add_distribution(self, dist: DistParams) -> MixedDistribution:
        self.distributions.append(dist)
        return self

    def rvs(self, size: int) -> np.ndarray:
        total_feature_len = np.sum([d.feature_len for d in self.distributions])
        result = np.zeros((size, total_feature_len))
        for dist in self.distributions:
            result[:, dist.idx_features] = dist.sample(size)
        return result

    def compute_joint_p(self, x: np.ndarray) -> np.ndarray:
        probs = [dist.compute_probability(x) for dist in self.distributions]
        stacked_probs = np.hstack(probs)
        result = np.prod(stacked_probs, -1)
        if np.any(result > 1) and DEBUG_DISTRIBUTION:
            print('Stop')
            probs = [dist.compute_probability(x) for dist in self.distributions]
        return result

    def __repr__(self):
        return f"@{type(self).__name__}[Distributions: {len(self.distributions)}]"


class ApproximateNormalDistribution(Dist):
    def init(self, **kwargs) -> ApproximateNormalDistribution:
        self.cov_mask = self.params.cov_mask
        self.event = self.params.key
        dist = None
        eps = kwargs.pop('eps', 0.1)  # TODO: EPS needs to be passed through
        str_event = f"Event {self.event:02d}"
        dim = self.params.dim

        if dim == 0:
            i = 2
            j = 5
            dist = stats.multivariate_normal(self.params._mean, self.params._cov, allow_singular=True)
            state, is_error, str_result = self._process_dist_result(dist, str_event, i, j)
            print(str_result)
            if not is_error:
                self.dist, self.state = dist, state
                return self

        cov = self.params.cov
        mean = self.params.mean
        no_eps = np.zeros_like(cov)
        diagonal_eps = np.identity(dim) * eps
        everywhere_eps = np.ones_like(cov) * eps
        for i, eps in enumerate([no_eps, diagonal_eps, everywhere_eps]):
            cov_unchanged = self.params.cov + eps
            # for j, cov in enumerate([cov_unchanged, cov_filled]):
            for j, cov in enumerate([cov_unchanged]):
                dist = self._create_multivariate(mean, cov)
                state, is_error, str_result = self._process_dist_result(dist, str_event, i, j)
                print(str_result)
                if not is_error:
                    self.dist, self.state = dist, state
                    return self

        state = ApproximationLevel(2, 4)
        print(f"WARNING {str_event}: Could not any approx for event {self.event} -- {state}")
        self.dist, self.state = stats.multivariate_normal(np.zeros_like(self.params.mean)), state
        return self

    def _process_dist_result(self, dist, str_event, i, j):
        state = ApproximationLevel(i, j)
        is_error = (type(dist) == FallbackableException)
        str_result = f"WARNING {str_event}: {state} -- Could not create because {dist}" if is_error else f"SUCCESS {str_event}: {state}"
        return state, is_error, str_result

    def _create_multivariate(self, mean: np.ndarray, cov: np.ndarray) -> Union[FallbackableException, ScipyDistribution]:

        try:
            return stats.multivariate_normal(mean, cov)
        except LinAlgError as e:
            return FallbackableException(e)
        except ValueError as e:
            if e.args[0] == "the input matrix must be positive semidefinite":
                return FallbackableException(e)
            else:
                raise e

    def compute_joint_p(self, x: np.ndarray) -> np.ndarray:
        if self.params.dim == 0:
            return self.dist.compute_probability(x)
        variables = np.diag(self.params.cov_mask)
        constants = ~variables
        const_mean = self.params._mean[constants]

        x_variables = x[:, variables]
        x_constants = x[:, constants]

        probs = self.dist.compute_probability(x_variables)
        # Checks if all follow exact constant distribution. Otherwise it's a violation and thus 0 probability.
        close_to_mean = np.isclose(const_mean[None], x_constants)
        not_deviating = np.all(close_to_mean, axis=1)

        # Multiplies whether constant was hit for each case and then returning their probability
        return probs * not_deviating

    def rvs(self, size=1) -> np.ndarray:
        if self.params.dim == 0:
            return self.dist.rvs(size)
        variables = np.diag(self.params.cov_mask)
        constants = ~variables
        const_mean = self.params._mean[constants]

        result = np.zeros((size, len(np.diag(self.cov_mask))))
        samples = self.dist.rvs(size)

        result[:, variables] = samples
        result[:, constants] = const_mean

        return result

    @property
    def mean(self):
        return self.dist.mean

    @property
    def cov(self):
        return self.dist.cov

    def __repr__(self):
        return f"@{type(self).__name__}[Fallback {self.approximation_level} - Mean: {self.mean}]"


class FallbackDist(Dist):
    def __init__(self, feature_len: int):
        self.feature_len = feature_len

    def compute_joint_p(self, x: np.ndarray):
        return np.zeros((len(x), ))

    @property
    def mean(self):
        return None

    @property
    def cov(self):
        return None

    def rvs(self, size: int) -> np.ndarray:
        return np.zeros((size, ))


class PFeaturesGivenActivity():
    def __init__(self, all_dists: Dict[int, DistParams], fallback: Dist):
        self.fallback = fallback
        self.all_dists = all_dists

    def __getitem__(self, key) -> DistParams:
        if key in self.all_dists:
            return self.all_dists.get(key)
        return self.fallback

    def __repr__(self):
        return f"@{type(self).__name__}[{self.all_dists}]"


class UnigramTransitionProbability(TransitionProbability):
    def init(self):
        events_slided = sliding_window(self.events, 2)
        self.trans_count_matrix: np.ndarray = np.zeros((self.vocab_len, self.vocab_len))
        self.trans_probs_matrix: np.ndarray = np.zeros((self.vocab_len, self.vocab_len))

        self.df_tra_counts = pd.DataFrame(events_slided.reshape((-1, 2)).tolist()).value_counts()
        self.trans_idxs = np.array(self.df_tra_counts.index.tolist(), dtype=int)
        self.trans_from = self.trans_idxs[:, 0]
        self.trans_to = self.trans_idxs[:, 1]
        self.trans_counts = np.array(self.df_tra_counts.values.tolist(), dtype=int)
        self.trans_count_matrix[self.trans_from, self.trans_to] = self.trans_counts
        self.trans_probs_matrix = self.trans_count_matrix / self.trans_count_matrix.sum(axis=1, keepdims=True)
        self.trans_probs_matrix[np.isnan(self.trans_probs_matrix)] = 0

        self.start_count_matrix = np.zeros((self.vocab_len, 1))
        self.start_events = self.events[:, 0]
        self.start_counts_counter = Counter(self.start_events)
        self.start_indices = np.array(list(self.start_counts_counter.keys()), dtype=int)
        self.start_counts = np.array(list(self.start_counts_counter.values()), dtype=int)
        self.start_count_matrix[self.start_indices, 0] = self.start_counts
        self.start_probs = self.start_count_matrix / self.start_counts.sum()

    def _compute_p_seq(self, events: np.ndarray) -> np.ndarray:
        flat_transistions = self.extract_transitions(events)
        probs = self.extract_transitions_probs(events.shape[0], flat_transistions)
        start_events = np.array(list(events[:, 0]), dtype=int)
        start_event_prob = self.start_probs[start_events, 0, None]
        return np.hstack([start_event_prob, probs])

    def extract_transitions_probs(self, num_events: int, flat_transistions: np.ndarray) -> np.ndarray:
        t_from = flat_transistions[:, 0]
        t_to = flat_transistions[:, 1]
        probs = self.trans_probs_matrix[t_from, t_to].reshape(num_events, -1)
        return probs

    def extract_transitions(self, events: np.ndarray) -> np.ndarray:
        events_slided = sliding_window(events, 2)
        events_slided_flat = events_slided.reshape((-1, 2))
        transistions = np.array(events_slided_flat.tolist(), dtype=int)
        return transistions

    def sample(self, sample_size: int) -> np.ndarray:
        # https://stackoverflow.com/a/40475357/4162265
        result = np.zeros((sample_size, self.max_len))
        pos_probs = np.repeat(self.start_probs.T, sample_size, axis=0)
        order_matrix = np.ones((sample_size, self.vocab_len)) * np.arange(0, self.vocab_len)[None]

        for curr_pos in range(self.max_len):
            starting_events = matrix_sample(pos_probs)
            result[..., curr_pos] = starting_events[..., 0]
            pos_matrix = (order_matrix == starting_events) * 1
            pos_probs = pos_matrix @ self.trans_probs_matrix
        return result

    def __call__(self, xt: np.ndarray, xt_prev: np.ndarray) -> np.ndarray:
        probs = self.trans_probs_matrix[xt_prev, xt]
        return probs


# class FaithfulEmissionProbability(ProbabilityMixin, ABC):


class EmissionProbability(ProbabilityMixin, ABC):
    def init(self) -> EmissionProbability:
        num_seq, seq_len, num_features = self.features.shape
        self.eps = 0.1
        self.events = self.events
        self.features = self.features
        events_flat = self.events.reshape((-1, ))
        features_flat = self.features.reshape((-1, num_features))
        sort_indices = events_flat.argsort()
        events_sorted = events_flat[sort_indices]
        features_sorted = features_flat[sort_indices]
        self.df_ev_and_ft: pd.DataFrame = pd.DataFrame(features_sorted)
        self.data_groups: Dict[int, pd.DataFrame] = {}
        self.df_ev_and_ft["event"] = events_sorted.astype(int)
        self.dists = self.estimate_params(self.df_ev_and_ft)
        return self

    def _group_events(self, data: pd.DataFrame) -> DataFrameGroupBy:
        return data.groupby('event')

    def set_eps(self, eps=1) -> EmissionProbability:
        self.eps = eps
        return self

    def set_data_mapping(self, data_mapping: Dict) -> EmissionProbability:
        self.data_mapping = BetterDict(data_mapping)
        self.feature_len = sum([1 for cols in self.data_mapping.flatten().values() for _ in cols])
        return self

    def compute_probs(self, events: np.ndarray, features: np.ndarray, is_log=False) -> np.ndarray:
        num_seq, seq_len, num_features = features.shape
        events_flat = events.reshape((-1, )).astype(int)
        features_flat = features.reshape((-1, num_features))
        unique_events = np.unique(events_flat)
        emission_probs = np.zeros_like(events_flat, dtype=float)
        for ev in unique_events:
            # https://stats.stackexchange.com/a/331324
            ev_pos = events_flat == ev
            # if ev == 37: DELETE
            #     print("STOP")
            distribution: Dist = self.dists[ev]
            emission_probs[ev_pos] = distribution.compute_joint_p(features_flat[ev_pos])
            # distribution = self.gaussian_dists[ev]
            # emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])

        result = emission_probs.reshape((num_seq, -1))
        return np.log(result) if is_log else result

    def estimate_fallback(self, data: pd.DataFrame):
        support = len(data)
        dist = MixedDistribution(-1)
        for grp, vals in self.data_mapping.get("numericals").items():
            partial_dist = GaussianParams().set_data(data).set_idx_features(vals).set_key(-1).set_support(support)
            dist.add_distribution(partial_dist)
        for grp, vals in self.data_mapping.get("categoricals").items():
            partial_dist = BernoulliParams().set_data(data).set_idx_features(vals).set_key(-1).set_support(support)
            dist.add_distribution(partial_dist)
        for grp, vals in self.data_mapping.get("binaricals").items():
            partial_dist = BernoulliParams().set_data(data).set_idx_features(vals).set_key(-1).set_support(support)
            dist.add_distribution(partial_dist)
        return dist

    def estimate_params(self) -> PFeaturesGivenActivity:
        return PFeaturesGivenActivity({}, self.estimate_fallback())

    def sample(self, events: np.ndarray) -> np.ndarray:
        num_seq, seq_len = events.shape
        feature_len = self.feature_len
        events_flat = events.reshape((-1, )).astype(int)
        unique_events = np.unique(events_flat)
        features = np.zeros((events_flat.shape[0], feature_len), dtype=float)
        for ev in unique_events:
            # https://stats.stackexchange.com/a/331324
            ev_pos = events_flat == ev
            # if ev == 37: DELETE
            #     print("STOP")
            distribution: Dist = self.dists[ev]
            features[ev_pos] = distribution.rvs(size=ev_pos.sum())
        result = features.reshape((num_seq, seq_len, -1))
        return result


class EmissionProbIndependentFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(self.df_ev_and_ft)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = [v[0] for v in self.data_mapping.get('numericals').values()]

            gaussian = IndependentGaussianParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, vals in self.data_mapping.subset(["binaricals", "categoricals"]).flatten().items() for v in vals]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class EmissionProbGroupedDistFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(self.df_ev_and_ft)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = [v[0] for v in self.data_mapping.get('numericals').values()]

            gaussian = GaussianParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            for grp, vals in self.data_mapping.get("categoricals").items():
                multinoulli = MultinoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
                dist.add_distribution(multinoulli)
            for grp, vals in self.data_mapping.get("binaricals").items():
                bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
                dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


# class BernoulliMixtureEmissionProbability(EmissionProbability):
#  https://www.kaggle.com/code/allunia/uncover-target-correlations-with-bernoulli-mixture/notebook


class DistributionConfig(ConfigurationSet):
    def __init__(
        self,
        tprobs: TransitionProbability,
        eprobs: EmissionProbability,
    ):
        self.tprobs = tprobs
        self.eprobs = eprobs
        self._list: List[DistributionConfig] = [tprobs, eprobs]

    @staticmethod
    def registry(tprobs: List[TransitionProbability] = None, eprobs: List[EmissionProbability] = None, **kwargs) -> DistributionConfig:
        tprobs = tprobs or [UnigramTransitionProbability()]
        # eprobs = eprobs or [EmissionProbabilityMixedFeatures(), EmissionProbability(), EmissionProbIndependentFeatures()]
        # eprobs = eprobs or [EmissionProbGroupedDistFeatures()]
        eprobs = eprobs or [EmissionProbIndependentFeatures()]
        combos = it.product(tprobs, eprobs)
        result = [DistributionConfig(*cnf) for cnf in combos]
        return result

    def set_vocab_len(self, vocab_len: int, **kwargs) -> DistributionConfig:
        for distribution in self._list:
            distribution.set_vocab_len(vocab_len)
        return self

    def set_max_len(self, max_len: int, **kwargs) -> DistributionConfig:
        for distribution in self._list:
            distribution.set_max_len(max_len)
        return self

    def set_data(self, data: Cases, **kwargs) -> DistributionConfig:
        for distribution in self._list:
            distribution.set_data(data)
        return self

    def set_data_mapping(self, data_mapping: Dict) -> DistributionConfig:
        for distribution in self._list:
            distribution.set_data_mapping(data_mapping)
        return self

    def init(self, **kwargs) -> DistributionConfig:
        for distribution in self._list:
            distribution.init(**kwargs)
        return self


class DataDistribution(ConfigurableMixin):
    def __init__(self, data: Cases, vocab_len: int, max_len: int, data_mapping: Dict = None, config: DistributionConfig = None):
        events, features = data.cases
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.data_mapping = data_mapping
        self.events = events
        self.features = self.convert_features(features, self.data_mapping)
        data = Cases(self.events, self.features)
        self.config = config.set_data(data).set_data_mapping(self.data_mapping)
        self.tprobs = self.config.tprobs
        self.eprobs = self.config.eprobs

    def convert_features(self, features, data_mapping):
        features = features.copy()
        col_cat_features = [idx for data_type, cols in data_mapping.items() for cname, indices in cols.items() for idx in indices if data_type in ["categoricals", "binaricals"]]
        features[..., col_cat_features] = np.where(features[..., col_cat_features] == 0, 0 - FIX_BINARY_OFFSET, features[..., col_cat_features])
        features[..., col_cat_features] = np.where(features[..., col_cat_features] == FIX_BINARY_OFFSET, 0, features[..., col_cat_features])
        features[..., col_cat_features] = np.where(features[..., col_cat_features] == FIX_BINARY_OFFSET + 1, 1, features[..., col_cat_features])
        return features

    def init(self) -> DataDistribution:
        self.config = self.config.set_vocab_len(self.vocab_len).set_max_len(self.max_len).init()
        return self

    def compute_probability(self, data: Cases) -> Tuple[np.ndarray, np.ndarray]:
        events, features = data.cases
        events = events
        features = self.convert_features(features, self.data_mapping)
        transition_probs = self.tprobs.compute_probs(events, cummulative=False)
        emission_probs = self.eprobs.compute_probs(events, features, is_log=False)
        return transition_probs, emission_probs

    def sample(self, size: int = 1) -> Cases:
        sampled_ev = self.tprobs.sample(size)
        sampled_ft = self.eprobs.sample(sampled_ev)
        return Cases(sampled_ev, sampled_ft)

    def get_config(self) -> BetterDict:
        return super().get_config()

    def __len__(self) -> int:
        return len(self.events)