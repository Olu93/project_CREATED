from typing import Dict, Tuple
from numpy.typing import NDArray
import numpy as np


class Cases():
    def __init__(self, events: NDArray, features: NDArray, outcomes: NDArray = None):
        self._events = events
        self._features = features
        self._outcomes = outcomes
        self._len = len(self._events)
        self.num_cases, self.max_len, self.num_features = features.shape
        self._viability: NDArray = None

    def tie_all_together(self):
        return self

    def sort(self):
        ev, ft = self.data
        viability = self.viability_values
        ranking = np.argsort(viability)
        sorted_ev, sorted_ft = ev[ranking], ft[ranking]
        sorted_viability = viability[ranking]
        return Cases(sorted_ev, sorted_ft).set_viability(sorted_viability)

    def set_viability(self, viability_values: NDArray):
        if not (len(self.events) == len(viability_values)):
            ValueError(f"Number of fitness_vals needs to be the same as number of population: {len(self)} != {len(viability_values)}")
        self._viability = viability_values
        return self

    def get_topk(self, k: int):
        return

    # def __next__(self):
    #     events, features, outcomes = self.events, self.features, self.outcomes
    #     for i in range(len(self)-1):
    #         yield Cases(events[i:i + 1], features[i:i + 1], outcomes[i:i + 1])
    #     raise StopIteration

    # def __iter__(self):
    #     return next(self)

    def __iter__(self):
        events, features, outcomes = self.events, self.features, self.outcomes
        for i in range(len(self)):
            yield Cases(events[i:i + 1], features[i:i + 1], outcomes[i:i + 1])
        # raise StopIteration

    def __len__(self):
        return self._len

    def assert_viability_is_set(self, raise_error=False):

        if raise_error and (self._viability is None):
            raise ValueError(f"Viability values where never set: {self._viability}")

        return self._viability is not None

    @property
    def avg_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viability.mean()

    @property
    def max_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viability.max()

    @property
    def median_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return np.median(self._viability)

    @property
    def viability_values(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viability.copy()

    @property
    def data(self) -> Tuple[NDArray, NDArray]:
        return self._events.copy(), self._features.copy()

    @property
    def events(self):
        return self._events.copy() if self._events is not None else None

    @property
    def features(self):
        return self._features.copy() if self._features is not None else None

    @property
    def outcomes(self):
        return self._outcomes.copy() if self._outcomes is not None else None

    @outcomes.setter
    def outcomes(self, outcomes):
        self._outcomes = outcomes

    @property
    def size(self):
        return self._len


class Population(Cases):
    def __init__(self, events: NDArray, features: NDArray, outcomes: NDArray = None):
        super(Population, self).__init__(events, features, outcomes)
        self._survivor = None
        self._mutation = None

    def tie_all_together(self):
        return self

    def set_mutations(self, mutations: NDArray):
        if len(self.events) != len(mutations): f"Number of mutations needs to be the same as number of population: {len(self)} != {len(mutations)}"
        self._mutation = mutations
        return self

    def set_fitness_values(self, fitness_values: NDArray):
        self.set_viability(fitness_values)
        return self

    @staticmethod
    def from_cases(obj: Cases):
        return Population(obj.events, obj.features, obj.outcomes)

    @property
    def avg_fitness(self) -> NDArray:
        return self.avg_viability

    @property
    def max_fitness(self) -> NDArray:
        return self.max_viability

    @property
    def median_fitness(self) -> NDArray:
        return self.median_viability

    @property
    def fitness_values(self) -> NDArray:
        return self.viability_values.T[0]

    @property
    def mutations(self):
        if self._mutation is None: raise ValueError(f"Mutation values where never set: {self._mutation}")
        return self._mutation.copy()


class GeneratorResult(Cases):
    def __init__(self, events: NDArray, features: NDArray, outcomes: NDArray, viabilities: NDArray):
        super().__init__(events, features, outcomes)
        self.set_viability(viabilities)

    @classmethod
    def from_cases(cls, population: Cases):
        events, features = population.data
        result = cls(events.astype(float), features, population.outcomes, population.viability_values)
        return result

    def get_topk(self, top_k: int = 5):
        ev, ft = self.data
        viab = self.viability_values
        outc = self.outcomes

        ranking = np.argsort(viab, axis=0)
        topk_indices = ranking[-top_k:].flatten()

        ev_chosen, ft_chosen, outc_chosen, viab_chosen  = ev[topk_indices], ft[topk_indices], outc[topk_indices], viab[topk_indices]
        return GeneratorResult(ev_chosen, ft_chosen, outc_chosen, viab_chosen)