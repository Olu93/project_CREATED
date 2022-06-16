from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from thesis_viability.viability.viability_function import ViabilityMeasure

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from thesis_commons import random
from thesis_commons.modes import MutationMode

def remove_padding(data:Sequence[Sequence[int]], pad_id:int=0) -> Sequence[Sequence[int]]:
    result:Sequence[Sequence[int]] = []
    for row in data:
        indices = [idx for idx, elem in enumerate(row) if elem != pad_id]
        start = min(indices) if len(indices) else pad_id
        end = max(indices)+1  if len(indices) else len(row)
        subset = row[start:end]
        result.append(subset)
    return result


class Viabilities():
    class Measures(IntEnum):
        VIABILITY = 0
        SPARCITY = 1
        SIMILARITY = 2
        DATA_LLH = 3
        OUTPUT_LLH = 4
        MODEL_LLH = 5

    def __init__(self, num_counterfactuals, num_factuals):
        self._parts = np.zeros((6, num_counterfactuals, num_factuals))
        self.num_factuals = num_factuals
        self.num_counterfactuals = num_counterfactuals

    def get(self, index: Measures = Measures.VIABILITY) -> NDArray:
        return self._parts[index]

    def set_viability(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.VIABILITY] = val
        return self

    def set_mllh(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.MODEL_LLH] = val
        return self

    def set_sparcity(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.SPARCITY] = val
        return self

    def set_similarity(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.SIMILARITY] = val
        return self

    def set_dllh(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.DATA_LLH] = val
        return self

    def set_ollh(self, val: NDArray) -> Viabilities:
        self._parts[Viabilities.Measures.OUTPUT_LLH] = val
        return self

    def max(self, index: Measures = Measures.VIABILITY):
        return self._parts[index].max()

    def __add__(self, other: Viabilities):
        new_viab = Viabilities(self.num_counterfactuals + other.num_counterfactuals, self.num_factuals)
        new_viab._parts = np.concatenate((self._parts, other._parts), axis=1)
        return new_viab

    def __repr__(self) -> str:
        return f"{self._parts[Viabilities.Measures.VIABILITY]}"

    def __getitem__(self, key) -> Viabilities:

        selected = self._parts[:, key]
        sparcity, similarity, dllh, ollh, viab, llh = self.get_parts(selected)
        num_cfs, num_fas = selected.shape[-2], selected.shape[-1]
        return Viabilities(num_cfs, num_fas).set_sparcity(sparcity).set_similarity(similarity).set_dllh(dllh).set_ollh(ollh).set_viability(viab).set_mllh(llh)

    def get_parts(self, selected):
        sparcity, similarity, dllh, ollh, llh, viab = (
            selected[Viabilities.Measures.SPARCITY],
            selected[Viabilities.Measures.SIMILARITY],
            selected[Viabilities.Measures.DATA_LLH],
            selected[Viabilities.Measures.OUTPUT_LLH],
            selected[Viabilities.Measures.MODEL_LLH],
            selected[Viabilities.Measures.VIABILITY],
        )

        return sparcity, similarity, dllh, ollh, viab, llh

    def __copy__(self) -> Viabilities:
        return Viabilities.from_array(self._parts.copy())

    def copy(self) -> Viabilities:
        return self.__copy__()

    @staticmethod
    def from_array(parts: NDArray) -> Viabilities:
        sparcity, similarity, dllh, ollh, llh, viab = (
            parts[Viabilities.Measures.SPARCITY],
            parts[Viabilities.Measures.SIMILARITY],
            parts[Viabilities.Measures.DATA_LLH],
            parts[Viabilities.Measures.OUTPUT_LLH],
            parts[Viabilities.Measures.MODEL_LLH],
            parts[Viabilities.Measures.VIABILITY],
        )
        num_cf, num_fa = viab.shape
        return Viabilities(num_cf, num_fa).set_sparcity(sparcity).set_similarity(similarity).set_dllh(dllh).set_ollh(ollh).set_viability(viab).set_mllh(llh)

    @property
    def sparcity(self) -> NDArray:
        return self._parts[Viabilities.Measures.SPARCITY]

    @property
    def similarity(self) -> NDArray:
        return self._parts[Viabilities.Measures.SIMILARITY]

    @property
    def dllh(self) -> NDArray:
        return self._parts[Viabilities.Measures.DATA_LLH]

    @property
    def ollh(self) -> NDArray:
        return self._parts[Viabilities.Measures.OUTPUT_LLH]

    @property
    def viabs(self) -> NDArray:
        return self._parts[Viabilities.Measures.VIABILITY]

    @property
    def mllh(self) -> NDArray:
        return self._parts[Viabilities.Measures.MODEL_LLH]


# TODO: Remove everything that has to do with viabilities to simplify this class. Subclasses will handle the rest
# TODO: Consider calling this Log instead of Cases
class Cases():
    def __init__(self, events: NDArray, features: NDArray, likelihoods: NDArray = None, viabilities: Viabilities = None):
        self._events = events
        self._features = features
        self._likelihoods = likelihoods if likelihoods is not None else np.empty((len(events), 1))
        self._len = len(self._events)
        self.num_cases, self.max_len, self.num_features = features.shape
        self._viabilities: Viabilities = viabilities

    def tie_all_together(self) -> Cases:
        return self

    def sample(self, sample_size: int) -> Cases:
        chosen = self._get_random_selection(sample_size)
        ev, ft, llhs, _ = self.all
        return Cases(ev[chosen], ft[chosen], llhs[chosen])

    def set_viability(self, viabilities: Viabilities) -> Cases:
        if not (len(self.events) == len(viabilities.viabs)):
            ValueError(f"Number of fitness_vals needs to be the same as number of population: {len(self)} != {len(viabilities.viabs)}")
        self._viabilities = viabilities
        return self

    def get_topk(self, k: int):
        return

    def __add__(self, other: Cases):
        if other == None:
            return self
        result = (
            self.events,
            self.features,
            self.likelihoods,
            self._viabilities if self._viabilities is not None else None,
        )
        oresult = (
            other.events,
            other.features,
            other.likelihoods,
            other.viabilities if other.viabilities is not None else None,
        )
        new_ev = np.vstack((result[0], oresult[0]))
        new_ft = np.vstack((result[1], oresult[1]))
        new_llh = np.vstack((result[2], oresult[2]))
        new_viab = result[3] + oresult[3] if result[3] is not None and oresult[3] is not None else None
        return Cases(new_ev, new_ft, new_llh, new_viab)

    def __getitem__(self, key) -> Cases:
        len_cases = len(self)
        ev = self._events[key] if len_cases > 1 else self._events
        ft = self._features[key] if len_cases > 1 else self._features
        llh = None if self._likelihoods is None else self._likelihoods[key] if len_cases > 1 else self._likelihoods
        viab = None if self._viabilities is None else self._viabilities[key] if len_cases > 1 else self._viabilities
        return Cases(ev, ft, llh, viab)

    def __iter__(self) -> Cases:
        events, features, likelihoods, viabilities = self.events, self.features, self.likelihoods, self.viabilities

        for i in range(len(self)):
            llh = likelihoods[i:i + 1] if likelihoods is not None else None
            viab = viabilities[i:i + 1] if viabilities is not None else None
            yield Cases(events[i:i + 1], features[i:i + 1], llh, viab)
        # raise StopIteration

    def __len__(self):
        return self._len

    def _get_random_selection(self, sample_size: int):
        num_cases = len(self)
        chosen = random.choice(np.arange(num_cases), size=sample_size, replace=False)
        return chosen

    def assert_viability_is_set(self, raise_error=False):

        if raise_error and (self._viabilities is None):
            raise ValueError(f"Viability values where never set: {self._viabilities}")

        return self._viabilities is not None

    @property
    def avg_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viabilities.viabs.mean(axis=0)

    @property
    def max_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return self._viabilities.viabs.max(axis=0)

    @property
    def median_viability(self) -> NDArray:
        self.assert_viability_is_set(raise_error=True)
        return np.median(self._viabilities.viabs, axis=0)

    @property
    def cases(self) -> Tuple[NDArray, NDArray]:
        return self._events.copy(), self._features.copy()

    @property
    def trimmed_events(self) -> Tuple[NDArray, NDArray]:
        return remove_padding(self._events.copy().astype(int))

    @property
    def all(self) -> Tuple[NDArray, NDArray, NDArray, Viabilities]:
        result = (
            self.events,
            self.features,
            self.likelihoods,
            self._viabilities if self._viabilities is not None else None,
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
    def outcomes(self):
        return self._likelihoods.copy() > 0.5 if self._likelihoods is not None else None

    @property
    def viabilities(self) -> Viabilities:
        return self._viabilities.copy() if self._viabilities is not None else None

    def set_likelihoods(self, likelihoods):
        self._likelihoods = likelihoods
        return self

    @property
    def size(self):
        return self._len

    def __repr__(self):
        results_string = "{"
        results_string += f"{type(self).__name__}| Ev:{self._events.shape} -- Ft:{self._features.shape}"
        if self._likelihoods is not None:
            results_string += f" -- LLH:{self._likelihoods.shape}"
        if self._viabilities is not None:
            results_string += f" -- VIAB:{self._viabilities.viabs.shape}"
        results_string += "}"
        return results_string


class EvaluatedCases(Cases):
    def __init__(self, events: NDArray, features: NDArray, viabilities: Viabilities = None):
        assert type(viabilities) in [Viabilities, type(None)], f"Vabilities is not the correct class {type(viabilities)}"
        super().__init__(events, features, viabilities.mllh if viabilities is not None else None)
        self._viabilities = viabilities
        self.instance_num: int = None
        self.factuals: Cases = None

    def sample(self, sample_size: int) -> EvaluatedCases:
        chosen = super()._get_random_selection(sample_size)
        ev, ft = self.cases
        viabilities = self.viabilities
        return EvaluatedCases(ev[chosen], ft[chosen], viabilities[chosen])

    def sort(self) -> SortedCases:
        ev, ft = self.cases
        viab = self.viabilities
        ranking = np.argsort(viab.viabs, axis=0)
        topk_indices = ranking.flatten()
        viab_chosen = Viabilities.from_array(viab._parts[:, topk_indices])
        ev_chosen, ft_chosen = ev[topk_indices], ft[topk_indices]
        return SortedCases(ev_chosen, ft_chosen, viab_chosen).set_ranking(np.argsort(viab_chosen.viabs, axis=0)[::-1] + 1)

    @classmethod
    def from_cases(cls, population: Cases):
        events, features = population.cases
        result = cls(events.astype(float), features, population.likelihoods, population.viabilities)
        return result

    def get_topk(self, top_k: int = 5):
        sorted_cases = self.sort()
        ev_chosen = sorted_cases._events[-top_k:]
        ft_chosen = sorted_cases._features[-top_k:]
        viab_chosen = sorted_cases._viabilities[-top_k:]
        # ranking_chosen = np.argsort(viab_chosen.viabs, axis=0)[::-1]+1
        ranking_chosen = np.argsort(viab_chosen.viabs, axis=0)[::-1] + 1
        return SortedCases(ev_chosen, ft_chosen, viab_chosen).set_ranking(ranking_chosen)

    def set_instance_num(self, num: int) -> EvaluatedCases:
        self.instance_num = num
        return self

    def set_creator(self, creator: str) -> EvaluatedCases:
        self.creator = creator
        return self

    def set_fa_case(self, factuals: Cases) -> EvaluatedCases:
        self.factuals = factuals
        return self

    def to_dict_stream(self):
        cf_trimmed_events = self.trimmed_events
        factual = self.factuals
        trimmed_factual_events = factual.trimmed_events[0]
        fa_outcome = factual.outcomes[0][0] * 1
        
        for i in range(len(self)):
            # cf_events = self._events[i].astype(int)
            yield i, {
                "creator": self.creator,
                "instance_num": self.instance_num,
                "cf_events": cf_trimmed_events[i],
                # "cf_features": self._features[i],
                "fa_events": trimmed_factual_events,
                # "fa_features": factual.features[0],
                "fa_outcome": fa_outcome,
                "likelihood": self._likelihoods[i][0],
                "outcome": ((self._likelihoods[i] > 0.5) * 1)[0],
                "viability": self._viabilities.viabs[i][0],
                "sparcity": self._viabilities.sparcity[i][0],
                "similarity": self._viabilities.similarity[i][0],
                "dllh": self._viabilities.dllh[i][0],
                "ollh": self._viabilities.ollh[i][0],
            }


class SortedCases(EvaluatedCases):
    def set_ranking(self, ranking) -> SortedCases:
        self.ranking = ranking
        return self

    def to_dict_stream(self):
        for i, case in super().to_dict_stream():
            yield {**case, 'rank': self.ranking[i][0]}


# TODO: Consider calling this Population instead of MutatedCases
class MutatedCases(EvaluatedCases):
    def __init__(self, events: NDArray, features: NDArray, viabilities: NDArray = None):
        super(MutatedCases, self).__init__(events, features, viabilities)

    def set_mutations(self, mutations: NDArray) -> MutatedCases:
        if len(self.events) != len(mutations): f"Number of mutations needs to be the same as number of population: {len(self)} != {len(mutations)}"
        self._mutation = mutations
        return self

    def evaluate_fitness(self, fitness_function: ViabilityMeasure, fa_seed: Cases) -> MutatedCases:
        fitness = fitness_function(fa_seed, self)
        return self.set_likelihoods(fitness.mllh).set_viability(fitness)

    @property
    def mutations(self):
        if self._mutation is None: return np.array([[MutationMode.NONE] * self._len])
        return self._mutation.copy()


class MutationRate():
    def __init__(self, p_delete: float = 0, p_insert: float = 0, p_change: float = 0, p_swap: float = 0, p_none: float = 0) -> None:
        num_mutation_types = len(MutationMode)
        self.probs = np.zeros(num_mutation_types)
        self.probs[MutationMode.DELETE] = p_delete
        self.probs[MutationMode.INSERT] = p_insert
        self.probs[MutationMode.CHANGE] = p_change
        self.probs[MutationMode.TRANSP] = p_swap
        self.probs[MutationMode.NONE] = p_none
        if p_delete + p_insert + p_change + p_swap + p_none == 0:
            self.probs = np.ones(num_mutation_types) / num_mutation_types
        if np.sum(self.probs) != 1:
            remaining = 1 - np.sum(self.probs)
            if remaining > 0:
                self.probs[MutationMode.NONE] += remaining  # Fill up
            else:
                raise Exception("Not a valid probability distribution")

    def to_dict(self):
        return {mode: self.probs[mode] for mode in MutationMode}

    def __repr__(self):
        return f"{self.to_dict()}"


class ConfigurableMixin(ABC):
    @abstractmethod
    def get_config(self):
        return {}


class Configuration(ConfigurableMixin):
    def __init__(self):
        self.name = type(self).__name__

    def set_name(self, name) -> Configuration:
        self.name = name
        return self

    def set_vocab_len(self, vocab_len) -> Configuration:
        self.vocab_len = vocab_len
        return self

    def set_sample_size(self, sample_size: int) -> Configuration:
        self.sample_size = sample_size
        return self

    @abstractmethod
    def get_config(self):
        return {
            'common': {
                "vocab_len": self.vocab_len,
                "num_population": self.sample_size,
            }
        }


class ConfigurationSet:
    list_of_configurations: List[Configuration] = []

    def append(self, configuration:Configuration) -> ConfigurationSet:
        self.list_of_configurations.append(configuration)
        return self
    
    def extend(self, list_configs:Configuration) -> ConfigurationSet:
        self.list_of_configurations.extend(list_configs)
        return self

    def get_config(self) -> Dict:
        result = {}
        for configuration in self.list_of_configurations:
            result.update(configuration.get_config())
        return result