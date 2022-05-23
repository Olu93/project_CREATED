from enum import IntEnum, auto
from tokenize import Number
from typing import Any, Counter, Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from thesis_commons.modes import MutationMode
from thesis_generators.models.model_commons import BaseModelMixin
from tqdm import tqdm
from thesis_viability.viability.viability_function import ViabilityMeasure
from numpy.typing import NDArray

DEBUG_STOP = 1000


class IterationStatistics():
    def __init__(self, num_instance: int) -> None:
        self.base_store = {}
        self.complex_store = {}
        self.num_instance: int = num_instance
        self.base_store["num_instance"] = num_instance

    # num_generation, num_population, num_survivors, fitness_values
    def update_base(self, key: str, val: Number):
        self.base_store[key] = val

    def update_mutations(self, key: str, mutations: Union[List[MutationMode], List[Sequence[MutationMode]]]):
        cnt = Counter((tuple(row) for row in mutations))
        self.complex_store[key] = cnt

    def __repr__(self):
        dict_copy = dict(self.base_store)
        return f"Instance {dict_copy.pop('num_instance')} {repr(dict_copy)}"


class GlobalStatistics():
    def __init__(self) -> None:
        self.store: List[IterationStatistics] = []
        self._stats:pd.DataFrame = None

    def attach(self, iteration_stats: IterationStatistics):
        self.store.append(iteration_stats)

    def compute(self, selection: List[int] = None):
        selected_stats = [stats for stats in self.store] if selection is None else [stats for stats in self.store if stats.num_instance in selection]

        base_stats = pd.DataFrame([stats.base_store for stats in selected_stats])
        complex_stats = pd.DataFrame([self._parse_complex(stats.complex_store) for stats in selected_stats])
        combined_stats = pd.concat([base_stats, complex_stats], axis=1)
        self._stats = combined_stats
        # self._stats = base_stats

        return self

    @property
    def stats(self, ) -> pd.DataFrame:
        return self._stats

    def _parse_complex(self, data: Dict[str, Any]):
        result = {f"{key}.{'_'.join(map(str, k))}":v for key, val in data.items() for k,v in val.items()} 
        return result


class Population():
    def __init__(self, events: NDArray, features: NDArray):
        self._events = events
        self._features = features
        self.num_cases, self.max_len, self.num_features = features.shape
        self._fitness = None
        self._survivor = None
        self._mutation = None

    def tie_all_together(self):
        return self

    def set_mutations(self, mutations: NDArray):
        assert len(self.events) == len(mutations), f"Number of mutations needs to be the same as number of population: {len(self)} != {len(mutations)}"
        self._mutation = mutations
        return self

    def set_fitness_vals(self, fitness_vals: NDArray):
        assert len(self.events) == len(fitness_vals), f"Number of fitness_vals needs to be the same as number of population: {len(self)} != {len(fitness_vals)}"
        self._fitness = fitness_vals
        return self

    def sort(self):
        ev, ft = self.items
        fitness = self.fitness_values
        ranking = np.argsort(fitness)
        sorted_ev, sorted_ft = ev[ranking], ft[ranking]
        sorted_fitness = fitness[ranking]
        return Population(sorted_ev, sorted_ft).set_fitness_vals(sorted_fitness)

    @property
    def avg_fitness(self) -> NDArray:
        assert self._fitness is not None, f"Fitness values where never set: {self._fitness}"
        return self._fitness.mean()

    @property
    def max_fitness(self) -> NDArray:
        assert self._fitness is not None, f"Fitness values where never set: {self._fitness}"
        return self._fitness.max()

    @property
    def median_fitness(self) -> NDArray:
        assert self._fitness is not None, f"Fitness values where never set: {self._fitness}"
        return np.median(self._fitness)

    @property
    def fitness_values(self) -> NDArray:
        assert self._fitness is not None, f"Fitness values where never set: {self._fitness}"
        return self._fitness.copy().T[0]

    @property
    def items(self):
        return self._events.copy(), self._features.copy()

    @property
    def events(self):
        return self._events.copy()

    @property
    def features(self):
        return self._features.copy()

    @property
    def mutations(self):
        assert self._mutation is not None, f"Mutation values where never set: {self._mutation}"
        return self._mutation.copy()

    def __len__(self):
        return len(self._events)

    @property
    def size(self):
        return len(self._events)


class EvolutionaryStrategy(BaseModelMixin, ABC):
    def __init__(self, evaluator: ViabilityMeasure, max_iter: int = 1000, survival_thresh: int = 5, num_population: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fitness_function = evaluator
        self.max_iter = max_iter
        self.name = self.__class__.__name__
        self.num_survivors = survival_thresh
        self.num_population = num_population
        self.evolutionary_counter = 0
        self.statistics = GlobalStatistics()
        self.curr_stats: IterationStatistics = None
        self.instance_pbar = None
        self.cycle_pbar = None
        self.results = {}

    def __call__(self, factual_seeds, labels):
        self.instance_pbar = tqdm(total=len(factual_seeds[0]))
        for instance_num, (fc_seed, fc_outcome) in enumerate(self.__next_seed(factual_seeds, labels)):
            self.curr_stats = IterationStatistics(instance_num)
            cf_parents = None
            self.cycle_pbar = tqdm(total=self.max_iter)
            cf_survivors = self.run_iteration(instance_num, self.evolutionary_counter, fc_seed, cf_parents)

            while not self.is_cycle_end(cf_survivors, self.evolutionary_counter, fc_seed):
                cf_survivors = self.run_iteration(instance_num, self.evolutionary_counter, fc_seed, cf_parents)
                cf_parents = cf_survivors

            # self.statistics
            final_population = cf_parents
            final_fitness = self.determine_fitness(final_population, fc_seed)
            self.results[instance_num] = (final_population, final_fitness)
            self.instance_pbar.update(1)
        self.statistics = self.statistics.compute()
        return self.results

    def run_iteration(self, instance_num: int, cycle_num: int, fc_seed: Population, cf_parents: Population):
        self.curr_stats.update_base("num_cycle", cycle_num)

        cf_offspring = self.generate_offspring(cf_parents, fc_seed)
        self.curr_stats.update_base("num_offspring", cf_offspring.size)
        self.curr_stats.update_mutations('mut_num_o', cf_offspring.mutations)

        cf_offspring = self.determine_fitness(cf_offspring, fc_seed)
        self.curr_stats.update_base("avg_offspring_fitness", cf_offspring.avg_fitness)

        cf_survivors = self.pick_survivors(cf_offspring)
        self.curr_stats.update_base("num_survivors", cf_survivors.size)
        self.curr_stats.update_base("avg_survivors_fitness", cf_survivors.avg_fitness)
        self.curr_stats.update_base("median_survivors_fitness", cf_survivors.median_fitness)
        self.curr_stats.update_base("max_survivors_fitness", cf_survivors.max_fitness)
        self.curr_stats.update_mutations('mut_num_s', cf_survivors.mutations)

        self.wrapup_cycle(instance_num)
        return cf_survivors

    def generate_offspring(self, cf_parents: Population, fc_seed: Population, **kwargs):
        if cf_parents is None:
            offspring = self._init_population(fc_seed)
            mutated = self._mutate_offspring(offspring, fc_seed)
            return mutated
        offspring = self._generate_population(cf_parents, fc_seed)
        mutated = self._mutate_offspring(offspring, fc_seed)
        return mutated

    def __next_seed(self, factual_seeds, labels) -> Tuple[Population, NDArray]:
        fc_events, fc_features = factual_seeds
        max_len = len(fc_events)
        for i in range(max_len):
            yield Population(fc_events[i][None, ...], fc_features[i][None, ...]), labels[i]

    @abstractmethod
    def _init_population(self, fc_seed: Population, **kwargs) -> Population:
        pass

    @abstractmethod
    def determine_fitness(self, cf_offspring: Population, fc_seed: Population, **kwargs) -> Population:
        pass

    @abstractmethod
    def _generate_population(self, cf_parents: Population, fc_seed: Population, **kwargs) -> Population:
        pass

    @abstractmethod
    def _recombine_parents(self, events, features, *args, **kwargs) -> Population:
        pass

    @abstractmethod
    def _mutate_offspring(self, cf_offspring: Population, fc_seed: Population, *args, **kwargs) -> Population:
        pass

    def pick_survivors(self, cf_offspring: Population, **kwargs) -> Population:
        cf_ev, cf_ft = cf_offspring.items
        fitness_values = cf_offspring.fitness_values
        mutations = cf_offspring.mutations
        ranking = np.argsort(fitness_values)
        selector = ranking[-self.num_survivors:]
        selected_fitness = fitness_values[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]
        selected_mutations = mutations[selector]
        selected = Population(selected_events, selected_features).set_fitness_vals(selected_fitness).set_mutations(selected_mutations)
        return selected

    def wrapup_cycle(self, instance_num: int, *args, **kwargs):
        self.evolutionary_counter += 1
        self.cycle_pbar.update(1)
        self.statistics.attach(self.curr_stats)
        self.curr_stats = IterationStatistics(instance_num)

    def is_cycle_end(self, *args, **kwargs) -> bool:
        return self.evolutionary_counter >= self.max_iter

    @property
    def stats(self):
        return self.statistics.stats

    # @abstractmethod
    # def __call__(self, *args, **kwargs):
    #     pass