from __future__ import annotations
from typing import Dict, Tuple
from numpy.typing import NDArray
import numpy as np


# TODO: Introduce static CaseBuilder class which builds all case types
# TODO: 
# TODO: Merge GeneratedResult with EvaluatedCases


class Cases():
    def __init__(self, events: NDArray, features: NDArray, likelihoods: NDArray = None):
        self._events = events
        self._features = features
        self._likelihoods = likelihoods
        self._len = len(self._events)
        self.num_cases, self.max_len, self.num_features = features.shape
        self._viabilities = None

    def tie_all_together(self) -> Cases:
        return self

    def sample(self, sample_size: int) -> Cases:
        chosen = self._get_random_selection(sample_size)
        ev, ft, llhs, _ = self.all
        return Cases(ev[chosen], ft[chosen], llhs[chosen])

    def set_viability(self, viabilities: NDArray) -> Cases:
        if not (len(self.events) == len(viabilities)):
            ValueError(f"Number of fitness_vals needs to be the same as number of population: {len(self)} != {len(viabilities)}")
        self._viabilities = viabilities
        return self

    def get_topk(self, k: int):
        return

    def __iter__(self) -> Cases:
        events, features, likelihoods = self.events, self.features, self.likelihoods
        for i in range(len(self)):
            yield Cases(events[i:i + 1], features[i:i + 1], likelihoods[i:i + 1])
        # raise StopIteration

    def __len__(self):
        return self._len

    def _get_random_selection(self, sample_size: int):
        num_cases = len(self)
        chosen = np.random.choice(np.arange(num_cases), size=sample_size, replace=False)
        return chosen

    def assert_viability_is_set(self, raise_error=False):

        if raise_error and (self._viabilities is None):
            raise ValueError(f"Viability values where never set: {self._viabilities}")

        return self._viabilities is not None

    @property
    def avg_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viabilities.mean()

    @property
    def max_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viabilities.max()

    @property
    def median_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return np.median(self._viabilities)



    @property
    def cases(self) -> Tuple[NDArray, NDArray]:
        return self._events.copy(), self._features.copy()

    @property
    def all(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        result = (
            self.events,
            self.features,
            self.likelihoods,
            self.viabilities,
        )
        return result

    @property
    def events(self):
        return self._events.copy() if self._events is not None else None

    @property
    def features(self):
        return self._features.copy() if self._features is not None else None

    @property
    def likelihoods(self):
        return self._likelihoods.copy() if self._likelihoods is not None else None

    @property
    def viabilities(self) -> NDArray:
        return self._viabilities.copy() if self._viabilities is not None else None

    def set_likelihoods(self, lieklihoods):
        self._likelihoods = lieklihoods
        return self

    @property
    def size(self):
        return self._len


class EvaluatedCases(Cases):
    def __init__(self, events: NDArray, features: NDArray, likelihoods: NDArray = None, viabilities: NDArray = None):
        super().__init__(events, features, likelihoods)
        self._viabilities = viabilities

    def sample(self, sample_size: int) -> EvaluatedCases:
        chosen = super()._get_random_selection(sample_size)
        ev, ft = self.cases
        viabilities = self.viabilities
        likelihoods = self.likelihoods
        return EvaluatedCases(ev[chosen], ft[chosen], likelihoods[chosen], viabilities[chosen])

    def sort(self) -> EvaluatedCases:
        ev, ft, _, viabs = self.all
        ranking = np.argsort(viabs)
        sorted_ev, sorted_ft = ev[ranking], ft[ranking]
        sorted_viability = viabs[ranking]
        return EvaluatedCases(sorted_ev, sorted_ft, None, sorted_viability)

# TODO: Rename to MutatedCases
class Population(EvaluatedCases):
    def __init__(self, events: NDArray, features: NDArray, likelihoods: NDArray = None, viabilities: NDArray = None):
        super(Population, self).__init__(events, features, likelihoods, viabilities)
        self._survivor = None
        self._mutation = None

    def set_mutations(self, mutations: NDArray):
        if len(self.events) != len(mutations): f"Number of mutations needs to be the same as number of population: {len(self)} != {len(mutations)}"
        self._mutation = mutations
        return self

    @property
    def mutations(self):
        if self._mutation is None: raise ValueError(f"Mutation values where never set: {self._mutation}")
        return self._mutation.copy()


class GeneratorResult(Cases):
    def __init__(self, events: NDArray, features: NDArray, likelihoods: NDArray, viabilities: NDArray):
        super().__init__(events, features, likelihoods)
        self.set_viability(viabilities)
        self.instance_num: int = None

    @classmethod
    def from_cases(cls, population: Cases):
        events, features = population.cases
        result = cls(events.astype(float), features, population.likelihoods, population.viabilities)
        return result

    def get_topk(self, top_k: int = 5):
        ev, ft = self.cases
        viab = self.viabilities
        outc = self.likelihoods

        ranking = np.argsort(viab, axis=0)
        topk_indices = ranking[-top_k:].flatten()

        ev_chosen, ft_chosen, outc_chosen, viab_chosen = ev[topk_indices], ft[topk_indices], outc[topk_indices], viab[topk_indices]
        return GeneratorResult(ev_chosen, ft_chosen, outc_chosen, viab_chosen)

    def set_instance_num(self, num: int) -> GeneratorResult:
        self.instance_num = num
        return self

    def set_creator(self, creator: str) -> GeneratorResult:
        self.creator = creator
        return self

    def to_dict_stream(self):
        for i in range(len(self)):
            yield {
                "creator": self.creator,
                "instance_num": self.instance_num,
                "events": self._events[i],
                "features": self._features[i],
                "likelihood": self._likelihoods[i][0],
                "outcome": ((self._likelihoods[i] > 0.5) * 1)[0],
                "viability": self._viabilities[i][0]
            }
