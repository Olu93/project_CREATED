from __future__ import annotations
from abc import ABC, abstractmethod
import itertools as it
from typing import TYPE_CHECKING, Callable, List, Tuple, Any, Dict, Sequence, Tuple, TypedDict
import sys
from thesis_commons.config import DEBUG_DISTRIBUTION, FIX_BINARY_OFFSET
from thesis_commons.functions import sliding_window
from thesis_commons.random import matrix_sample
from thesis_commons.representations import BetterDict, Cases, ConfigurableMixin, ConfigurationSet
from sklearn.preprocessing import StandardScaler

from thesis_readers.readers.AbstractProcessLogReader import FeatureInformation
if TYPE_CHECKING:
    pass

from collections import Counter, defaultdict
from ctypes import Union
from enum import IntEnum, auto
from thesis_commons.constants import CDType, CDomain
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.linalg import LinAlgError
from scipy.stats._distn_infrastructure import rv_frozen as ScipyDistribution
# from scipy.stats._discrete_distns import rv_discrete
from pandas.core.groupby.generic import DataFrameGroupBy

EPS = np.finfo(float).eps
EPS_BIG = 0.00001


def row_roll(arr, shifts, axis=1, fill=np.nan):
    # https://stackoverflow.com/a/65682885/4162265
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray, dtype int. Shape: `(arr.shape[:axis],)`.
        Amount to roll each row by. Positive shifts row right.

    axis : int
        Axis along which elements are shifted. 
        
    fill: bool or float
        If True, value to be filled at missing values. Otherwise just rolls across edges.
    """
    if np.issubdtype(arr.dtype, int) and isinstance(fill, float):
        arr = arr.astype(float)

    shifts2 = shifts.copy()
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]
    # Convert to a positive shift
    shifts2[shifts2 < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts2[:, np.newaxis]

    result = arr[tuple(all_idcs)]

    if fill is not False:
        # Create mask of row positions above negative shifts
        # or below positive shifts. Then set them to np.nan.
        *_, nrows, ncols = arr.shape

        mask_neg = shifts < 0
        mask_pos = shifts >= 0

        shifts_pos = shifts.copy()
        shifts_pos[mask_neg] = 0
        shifts_neg = shifts.copy()
        shifts_neg[mask_pos] = ncols + 1  # need to be bigger than the biggest positive shift
        shifts_neg[mask_neg] = shifts[mask_neg] % ncols

        indices = np.stack(nrows * (np.arange(ncols), ))
        nanmask = (indices < shifts_pos[:, None]) | (indices >= shifts_neg[:, None])
        result[nanmask] = fill

    arr = np.swapaxes(result, -1, axis)

    return arr


def strided_indexing_roll(a, r):
    # https://stackoverflow.com/a/65682885/4162265
    # Concatenate with sliced to cover all rolls
    p = np.full((a.shape[0], a.shape[1] - 1), np.nan)
    a_ext = np.concatenate((p, a, p), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), -r + (n - 1), 0]


def is_invertible(a):  # https://stackoverflow.com/a/17931970/4162265
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def is_singular(a):  # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    return np.linalg.cond(a) < 1 / sys.float_info.epsilon


# https://stats.stackexchange.com/questions/78177/posterior-covariance-of-normal-inverse-wishart-not-converging-properly
class NormalInverseWishartDistribution(object):
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / self.lmbda), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    foo[i, j] = np.sqrt(stats.chi2.rvs(self.nu - (i + 1) + 1))
                else:
                    foo[i, j] = np.random.normal(0, 1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

    def posterior(self, data):
        n = len(data)
        mean_data = np.mean(data, axis=0)
        sum_squares = np.sum([np.array(np.matrix(x - mean_data).T * np.matrix(x - mean_data)) for x in data], axis=0)
        mu_n = (self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
        lmbda_n = self.lmbda + n
        nu_n = self.nu + n
        psi_n = self.psi + sum_squares + self.lmbda * n / float(self.lmbda + n) * np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu))
        return NormalInverseWishartDistribution(mu_n, lmbda_n, nu_n, psi_n)


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

    def set_feature_info(self, feature_info: FeatureInformation) -> ProbabilityMixin:
        self.feature_info = feature_info
        return self


class TransitionProbability(ProbabilityMixin, ABC):
    def compute_probs(self, events, logdomain=False, joint=False, cummulative=True, **kwargs) -> np.ndarray:
        res = ResultTransitionProb(self._compute_p_seq(events, **kwargs))
        result = res.pnt_log_p
        if not (cummulative or joint or logdomain):
            return res.pnt_p
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
        # if ((data.max() != 0) or (data.min() != 0)) and DEBUG_DISTRIBUTION:
        #     print("STOP distributions.py")

        if not self.dist:
            return np.ones((len(data), 1))
        tmp = (data > 0) * 1
        result = self.dist.pmf(tmp)
        if np.any(np.isnan(result)) and DEBUG_DISTRIBUTION:
            print("STOP distributions.py")
        if (len(result.shape) > 2) and DEBUG_DISTRIBUTION:
            print("STOP distributions.py")
        return result

    def sample(self, size) -> np.ndarray:
        if not self.dist:
            return np.ones((size, self.feature_len)) * (0 - FIX_BINARY_OFFSET)
        return self.dist.rvs(size)


class ContinuousDistribution(DistParams):
    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        result = self.dist.pdf(data[:, self.idx_features]) if len(data) > 1 else np.array([self.dist.pdf(data[:, self.idx_features])])
        if np.any(np.isnan(result)) and DEBUG_DISTRIBUTION:
            print("STOP distributions.py")
        return result

    def sample(self, size) -> np.ndarray:
        return self.dist.rvs(size)


class BernoulliParams(DiscreteDistribution):
    def init(self) -> BernoulliParams:
        self._data: np.ndarray = self.data[self.idx_features]
        if np.all(np.unique(self._data) == (0 - FIX_BINARY_OFFSET)):
            return self
        self._counts = self._data.sum()
        self._p = self._counts / len(self._data)
        self.dist = stats.bernoulli(p=self._p)
        return self

    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        result = super().compute_probability(data[:, self.idx_features])
        if (len(result.shape) > 2) and DEBUG_DISTRIBUTION:
            print("STOP distributions.py")
        return result

    def sample(self, size) -> np.ndarray:
        if np.all(np.unique(self._data) == 0 - FIX_BINARY_OFFSET):
            sampled = np.ones((size, self.feature_len)) * (0 - FIX_BINARY_OFFSET)
            result = sampled * 1
            return result
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
        if self._p.size:
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
    eps = EPS_BIG

    def init(self) -> GaussianParams:
        # https://stackoverflow.com/a/35293215/4162265
        # self._scaler = StandardScaler()
        self._original_data = self.data[self.idx_features]
        # if self.key == -1:
        #     print("pause")

        # self._data = pd.DataFrame(self._scaler.fit_transform(self._original_data), columns=self._original_data.columns)
        self._data = self._original_data
        self._mean = self._data.mean().values
        self._cov = self._data.cov().values if self.support > 2 else np.zeros((self.feature_len, self.feature_len))
        self._var = np.nan_to_num(self._data.var().values, 1.0)
        self.cov_matrix_diag_indices = np.diag_indices_from(self._cov)
        self._cov[self.cov_matrix_diag_indices] = self._var
        # self.key = self.key
        # self.cov_mask = self.compute_cov_mask(self._cov)
        self.dist = self.create_dist()

        return self

    def create_dist(self):
        if (self._mean.sum() == 0) and (self._cov.sum() == 0):
            print(f"Activity {self.key}: Mean and Covariance are zero -- Use degenerate Gaussian")
            dist = stats.multivariate_normal(self._mean, self._cov, allow_singular=True)
            return dist

        dist = self.attempt_create_multivariate_gaussian(self._mean, self._cov, "1/4 Try proper distribution", allow_singular=False)
        if dist:
            return dist

        new_cov = self._cov.copy()
        tmp_diag = np.maximum(self._var, EPS_BIG)

        new_cov[self.cov_matrix_diag_indices] = tmp_diag
        dist = self.attempt_create_multivariate_gaussian(self.mean, new_cov, "2/4 Try partial diag constant addition", allow_singular=False)
        if dist:
            return dist

        new_cov = self._cov.copy()
        tmp = self._var + EPS_BIG
        new_cov[self.cov_matrix_diag_indices] = tmp
        dist = self.attempt_create_multivariate_gaussian(self._mean, new_cov, "3/4 Try full diag constant addition", allow_singular=False)
        if dist:
            return dist

        new_cov = self._cov.copy()
        new_cov[new_cov == 0] += EPS_BIG
        dist = self.attempt_create_multivariate_gaussian(self._mean, new_cov, "4/4 Try full matrix constant addition", allow_singular=False)
        if dist:
            return dist

        print(f"Activity {self.key}: {'Everything failed!'} -- Use degenerate Gaussian")
        dist = stats.multivariate_normal(self._mean, self._cov, allow_singular=True)
        self._cov_old = self._cov
        self._cov = new_cov
        return dist

        # try:
        #     new_cov = np.eye(*self._cov.shape) * EPS
        #     dist = stats.multivariate_normal(self._mean, new_cov)
        #     return dist
        # except Exception as e:
        #     print(f"Standard Normal Dist: Could not create Gaussian for Activity {self.key}: {e}")

    def attempt_create_multivariate_gaussian(self, mean, cov, attempt_text, allow_singular=False):
        dist = None
        exception = None
        try:
            dist = stats.multivariate_normal(mean, cov, allow_singular)
        except LinAlgError as e:
            exception = FallbackableException(f"Activity {self.key}: FAILURE -> {attempt_text} -- {e}")
            # return dist
        # except ValueError as e:
        #     if "singular matrix" in str(e):
        #         exception = FallbackableException(f"Activity {self.key}: FAILURE -> Gaussian creation -- {e}")
        # return dist

        if isinstance(exception, FallbackableException):
            print(exception)
            return dist

        if dist:
            print(f"Activity {self.key}: SUCCESS -> {attempt_text}")
            self._cov_old = self._cov
            self._cov = dist.cov
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
        self._cov: np.ndarray = np.diag(self._var) if self.support > 2 else np.zeros_like(self._cov)
        return self


class ChiSquareParams(GaussianParams):
    def init(self) -> ChiSquareParams:
        super().init()

    def _mahalanobis(self, delta, cov) -> np.ndarray:
        # https://stackoverflow.com/a/55095136/4162265
        ci = np.linalg.pinv(cov)
        return np.sum(((delta @ ci) * delta), axis=-1)

    def compute_probability(self, data: np.ndarray) -> np.ndarray:
        # https://stats.stackexchange.com/a/331324
        delta = data[:, self.idx_features] - self.dist.mean
        cov = self.dist.cov
        distance = self._mahalanobis(delta, cov)
        result = 1 - stats.chi2.cdf(distance, len(cov))
        return result[:, None]


class GaussianWithNormalPriorParams(ChiSquareParams):

    # https://stats.stackexchange.com/q/28744/361976
    def create_dist(self) -> GaussianWithNormalPriorParams:
        d = self._data.shape[1]
        n = self.support
        mu_0 = np.zeros(d)
        E_0 = np.eye(d)
        mu_n = self._mean.copy()
        E_n = self._cov.copy()
        lmbda_n = 1

        helper_dist = NormalInverseWishartDistribution(mu_0, n, d + 2, E_0).posterior(self._data)
        mean, cov = helper_dist.sample()
        dist = self.attempt_create_multivariate_gaussian(mean, cov, f"Activity {self.key}: Try bayesian gauss distribution", allow_singular=False)
        if dist:
            self._mean, self._cov = mean, cov
            return dist

        dist = self.attempt_create_multivariate_gaussian(self._mean, self._cov, f"Activity {self.key}: Try degenerate gauss distribution", allow_singular=True)
        return dist

    # def create_dist(self) -> GaussianWithNormalPriorParams:
    #     d = self._data.shape[1]
    #     n = self.support
    #     mu_0 = np.zeros(d)
    #     E_0 = np.eye(d)
    #     mu_n = self._mean.copy()
    #     E_n = self._cov.copy()

    #     E_adj = np.linalg.inv((E_0 + ((1/n) * E_n)))

    #     self._mean = E_0 @ E_adj @ self._data.mean(0) + ((1/n) * E_n) @ E_adj @ mu_0
    #     self._cov = (1/n) * (E_0 @ E_adj @ E_n)
    #     dist = self.attempt_create_multivariate_gaussian(self._mean, self._cov, f"Activity {self.key}: Try bayesian gauss distribution", allow_singular=False)
    #     if dist:
    #         return dist

    #     dist = self.attempt_create_multivariate_gaussian(self._mean, self._cov, f"Activity {self.key}: Try degenerate gauss distribution", allow_singular=True)
    #     return dist

    # https://stats.stackexchange.com/a/50902/361976
    # https://stats.stackexchange.com/a/78188/361976
    # https://handwiki.org/wiki/Normal-inverse-Wishart_distribution
    # def create_dist(self) -> GaussWithNormalPriorParams:
    #     d = self.feature_len
    #     n = self.support
    #     k_0 = 1/n
    #     v_0 = 1/n
    #     E_0 = np.eye(d)
    #     mu_0 = np.zeros(d)

    #     E_n = self._cov.copy()
    #     mu_n = self._mean.copy()

    #     E_adj = np.linalg.inv((E_0 + ((1/n) * E_n)))

    #     x_mean = self.data.mean(0)[:, None]
    #     pairwise_dev = (self.data.T - x_mean)
    #     psi = v_0 * E_0
    #     C = pairwise_dev@pairwise_dev.T


class FallbackableException(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


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
            result[:, dist.idx_features] = dist.sample(size).reshape(result[:, dist.idx_features].shape)  # Could break things
            if isinstance(dist, DiscreteDistribution):
                result[:, dist.idx_features] = result[:, dist.idx_features] + FIX_BINARY_OFFSET
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


class MarkovChainProbability(TransitionProbability):
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
        self.trans_probs_df = pd.DataFrame(self.trans_probs_matrix, index=range(self.vocab_len), columns=range(self.vocab_len))

        self.start_count_matrix = np.zeros((self.vocab_len, 1))
        self.start_events = self.events[:, 0]
        self.start_counts_counter = Counter(self.start_events)
        self.start_indices = np.array(list(self.start_counts_counter.keys()), dtype=int)
        self.start_counts = np.array(list(self.start_counts_counter.values()), dtype=int)
        self.start_count_matrix[self.start_indices, 0] = self.start_counts
        self.start_probs = self.start_count_matrix / self.start_counts.sum()

        self.end_count_matrix = np.zeros((self.vocab_len, 1))
        self.end_events = self.events[:, -1]
        self.end_counts_counter = Counter(self.end_events)
        self.end_indices = np.array(list(self.end_counts_counter.keys()), dtype=int)
        self.end_counts = np.array(list(self.end_counts_counter.values()), dtype=int)
        self.end_count_matrix[self.end_indices, 0] = self.end_counts
        self.end_probs = self.end_count_matrix / self.end_counts.sum()

    def _compute_p_seq(self, events: np.ndarray) -> np.ndarray:
        flat_transistions = self.extract_transitions(events)
        probs = self.extract_transitions_probs(events.shape[0], flat_transistions)
        start_events = np.array(list(events[:, 0]), dtype=int)
        start_event_prob = self.start_probs[start_events, 0, None]
        end_events = np.array(list(events[:, -1]), dtype=int)
        end_event_prob = self.end_probs[end_events, 0, None]
        end_event_prob[events.sum(-1) == 0] = 0
        spark = np.ones_like(end_event_prob, dtype=int)
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
        mask = np.ones((sample_size, self.max_len))
        end_events = np.array(list(self.end_counts_counter.keys()))
        order_matrix = np.ones((sample_size, self.vocab_len)) * np.arange(0, self.vocab_len)[None]

        # curr_pos = 0
        is_end = np.ones((sample_size, 1)) == 0
        pos_probs = np.repeat(self.start_probs.T, sample_size, axis=0)

        # next_event = matrix_sample(pos_probs)
        # result[..., curr_pos] = next_event[..., 0]
        # mask[..., curr_pos] = np.isin(next_event, end_events)[..., 0]
        # is_end = is_end | np.isin(next_event, end_events)
        # pos_matrix = (order_matrix == next_event) * 1
        # pos_probs = pos_matrix @ self.trans_probs_matrix

        for curr_pos in range(0, self.max_len):
            next_event = matrix_sample(pos_probs)
            result[..., curr_pos] = next_event[..., 0]
            mask[..., curr_pos] = is_end[..., 0]
            is_end = is_end | np.isin(next_event, end_events)
            pos_matrix = (order_matrix == next_event) * 1
            pos_probs = pos_matrix @ self.trans_probs_matrix

        rev_mask = (~(mask == 1))

        shifts = self.max_len - rev_mask.cumprod(-1).sum(-1).astype(int)
        result_with_removed_post_events = result * rev_mask
        result_rolled_to_end = row_roll(result_with_removed_post_events, shifts)
        results_fill_nan = np.nan_to_num(result_rolled_to_end, True, 0)
        return results_fill_nan

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

    def set_feature_info(self, feature_info: FeatureInformation) -> EmissionProbability:
        self.feature_info = feature_info
        self.feature_len = feature_info.ft_len
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
            #     print("STOP distributions.py")
            distribution: Dist = self.dists[ev]
            emission_probs[ev_pos] = distribution.compute_joint_p(features_flat[ev_pos])
            # distribution = self.gaussian_dists[ev]
            # emission_probs[ev_pos] = distribution.pdf(features_flat[ev_pos])

        result = emission_probs.reshape((num_seq, -1))
        return np.log(result) if is_log else result

    def estimate_fallback(self, data: pd.DataFrame):
        support = len(data)
        dist = MixedDistribution(-1)
        for grp, vals in self.feature_info.idx_continuous.items():
            partial_dist = GaussianParams().set_data(data).set_idx_features([vals]).set_key(-1).set_support(support)
            dist.add_distribution(partial_dist)
        for grp, vals in self.feature_info.idx_discrete.items():
            partial_dist = BernoulliParams().set_data(data).set_idx_features([vals]).set_key(-1).set_support(support)
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
            #     print("STOP distributions.py")
            distribution: Dist = self.dists[ev]
            features[ev_pos] = distribution.rvs(size=ev_pos.sum())
        result = features.reshape((num_seq, seq_len, -1))
        return result


class EmissionProbIndependentFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = IndependentGaussianParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class DefaultEmissionProbFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = GaussianParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class ChiSqEmissionProbFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = ChiSquareParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class EmissionProbGroupedDistFeatures(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = GaussianWithNormalPriorParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class BayesianDistFeatures1(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = GaussianWithNormalPriorParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
            bernoulli = BernoulliParams().set_data(df).set_idx_features(vals).set_key(activity).set_support(support)
            dist.add_distribution(bernoulli)
            all_dists[activity] = dist.init()
        print("Create P(ft|ev=None)")
        fallback = self.estimate_fallback(original_data.drop('event', axis=1)).init()
        return PFeaturesGivenActivity(all_dists, fallback)


class BayesianDistFeatures2(EmissionProbability):
    def estimate_params(self, data: pd.DataFrame):
        original_data = data.copy()
        data = self._group_events(original_data)
        all_dists = {}
        print("Create P(ft|ev=X)")
        for activity, df in data:
            support = len(df)
            dist = MixedDistribution(activity)
            idx_features = list(self.feature_info.idx_continuous.values())

            gaussian = GaussianWithNormalPriorParams().set_data(df).set_idx_features(idx_features).set_key(activity).set_support(support)
            dist = dist.add_distribution(gaussian)
            vals = [v for grp, v in self.feature_info.idx_discrete.items()]
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
        tprobs = tprobs or [MarkovChainProbability()]
        # eprobs = eprobs or [EmissionProbabilityMixedFeatures(), EmissionProbability(), EmissionProbIndependentFeatures()]
        # eprobs = eprobs or [DefaultEmissionProbFeatures()]
        # eprobs = eprobs or [EmissionProbability()]
        # eprobs = eprobs or [ChiSqEmissionProbFeatures()]
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

    def set_feature_info(self, feature_info: FeatureInformation) -> DistributionConfig:
        for distribution in self._list:
            distribution.set_feature_info(feature_info)
        return self

    def init(self, **kwargs) -> DistributionConfig:
        for distribution in self._list:
            distribution.init(**kwargs)
        return self


class DataDistribution(ConfigurableMixin):
    def __init__(self, data: Cases, vocab_len: int, max_len: int, feature_info: FeatureInformation = None, config: DistributionConfig = None):
        self.original_data = data
        events, features = data.cases
        self.vocab_len = vocab_len
        self.max_len = max_len
        self.feature_info = feature_info
        self.events = events
        self.features = self.convert_features(features, self.feature_info)
        data = Cases(self.events, self.features)
        self.config = config.set_data(data).set_feature_info(self.feature_info)
        self.tprobs = self.config.tprobs
        self.eprobs = self.config.eprobs
        self.pad_value = 0 - FIX_BINARY_OFFSET

    def convert_features(self, features, feature_info: FeatureInformation):
        features = features.copy()
        col_cat_features = list(feature_info.idx_discrete.values())
        # features[..., col_cat_features] = np.where(features[..., col_cat_features] == 0, 0 - FIX_BINARY_OFFSET, features[..., col_cat_features])
        # features[..., col_cat_features] = np.where(features[..., col_cat_features] == FIX_BINARY_OFFSET, 0, features[..., col_cat_features])
        # features[..., col_cat_features] = np.where(features[..., col_cat_features] == FIX_BINARY_OFFSET + 1, 1, features[..., col_cat_features])
        features[..., col_cat_features] = features[..., col_cat_features] - FIX_BINARY_OFFSET
        return features

    def init(self) -> DataDistribution:
        self.config = self.config.set_vocab_len(self.vocab_len).set_max_len(self.max_len).init()
        return self

    def compute_probability(self, data: Cases) -> Tuple[np.ndarray, np.ndarray]:
        events, features = data.cases
        events = events
        features = self.convert_features(features, self.feature_info)
        # events = np.vstack([events, np.zeros_like(events)[-1]])
        # features = np.vstack([features, np.zeros_like(features)[-1, None]])
        transition_probs = self.tprobs.compute_probs(events, cummulative=False)
        emission_probs = np.minimum(self.eprobs.compute_probs(events, features, is_log=False), 1)
        if np.array(emission_probs > 1).any() & DEBUG_DISTRIBUTION:
            problematic = np.unique(np.where(emission_probs > 1)[0])
            self.eprobs.compute_probs(events, features, is_log=False)
        return transition_probs, emission_probs

    def sample(self, size: int = 1) -> Cases:
        sampled_ev = self.tprobs.sample(size)
        sampled_ft = self.eprobs.sample(sampled_ev)
        return Cases(sampled_ev, sampled_ft)

    def sample_features(self, events: np.ndarray) -> np.ndarray:
        sampled_ft = self.eprobs.sample(events)
        return sampled_ft

    def get_config(self) -> BetterDict:
        return BetterDict(super().get_config()).merge({
            'type': type(self).__name__,
            'name': f"{type(self.tprobs).__name__}_{type(self.eprobs).__name__}",
            'transition_estimator': type(self.tprobs).__name__,
            'emission_estimator': type(self.eprobs).__name__,
        })

    def __len__(self) -> int:
        return len(self.events)

        # next_event = matrix_sample(pos_probs)
        # result[..., curr_pos] = next_event[..., 0]
        # mask[..., curr_pos] = is_end[..., 0]
        # is_end = is_end | np.isin(next_event, end_events)
        # pos_matrix = (order_matrix == next_event) * 1
        # pos_probs = pos_matrix @ self.trans_probs_matrix

        # for curr_pos in range(1,self.max_len):
        #     next_event = matrix_sample(pos_probs)
        #     result[..., curr_pos] = next_event[..., 0]
        #     mask[..., curr_pos] = is_end[..., 0]
        #     is_end = is_end | np.isin(next_event, end_events)
        #     pos_matrix = (order_matrix == next_event) * 1
        #     pos_probs = pos_matrix @ self.trans_probs_matrix