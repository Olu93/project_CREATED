from abc import ABC, abstractmethod
from tokenize import Number
from typing import Any, Counter, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis_commons.model_commons import BaseModelMixin
from thesis_commons.modes import MutationMode
from thesis_commons.representations import Cases, MutatedCases
from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG_STOP = 1000


class IterationStatistics():
    def __init__(self) -> None:
        self.base_store = {}
        self.complex_store = {}

    # num_generation, num_population, num_survivors, fitness_values
    def update_base(self, key: str, val: Number):
        self.base_store[key] = val

    def update_mutations(self, key: str, mutations: Union[List[MutationMode], List[Sequence[MutationMode]]]):
        cnt = Counter((tuple(row) for row in mutations))
        self.complex_store[key] = cnt

    def __repr__(self):
        dict_copy = dict(self.base_store)
        return f"Iteration {dict_copy.pop('num_instance')} {repr(dict_copy)}"


class InstanceStatistics():
    def __init__(self) -> None:
        self.store: List[IterationStatistics] = []
        self._stats: pd.DataFrame = None

    def attach(self, iteration_stats: IterationStatistics):
        self.store.append(iteration_stats)

    def compute(self, selection: List[int] = None):
        selected_stats = [stats for stats in self.store] if selection is None else [stats for stats in self.store if stats.num_instance in selection]

        base_stats = pd.DataFrame([stats.base_store for stats in selected_stats])
        complex_stats = pd.DataFrame([self._parse_complex(stats.complex_store) for stats in selected_stats]).fillna(0)
        combined_stats = pd.concat([base_stats, complex_stats], axis=1)
        self._stats = combined_stats
        # self._stats = base_stats

        return self

    @property
    def stats(self, ) -> pd.DataFrame:
        return self._stats

    def _parse_complex(self, data: Dict[str, Any]):
        result = {f"{key}.{'_'.join(map(str, k))}": v for key, val in data.items() for k, v in val.items()}
        return result

# TODO: Rename num_population to sample size
# TODO: Actually remove num_population and put it to predict
class EvolutionaryStrategy(BaseModelMixin, ABC):
    def __init__(self, evaluator: ViabilityMeasure, max_iter: int = 1000, survival_thresh: int = 25, num_population: int = 100, **kwargs) -> None:
        super(EvolutionaryStrategy, self).__init__(**kwargs)
        self.fitness_function = evaluator
        self.max_iter:int = max_iter
        self.name:str = self.__class__.__name__
        self.num_survivors: int = survival_thresh
        self.num_population: int = num_population
        self.num_cycle: int = 0
        self.statistics: InstanceStatistics = InstanceStatistics()
        self.curr_stats: IterationStatistics = None
        self.cycle_pbar: tqdm = None
        # self._stats: Sequence[IterationStatistics] = []

    def predict(self, fa_case: Cases, **kwargs) -> Tuple[MutatedCases, InstanceStatistics]:
        fa_seed = Cases(*fa_case.all)
        self.curr_stats = IterationStatistics()
        cf_parents: MutatedCases = None
        self.num_cycle = 0
        self.cycle_pbar = tqdm(total=self.max_iter, desc="Evo Cycle")
        cf_survivors = self.run_iteration(self.num_cycle, fa_seed, cf_parents)
        self.wrapup_cycle()

        while not self.is_cycle_end(cf_survivors, self.num_cycle, fa_seed):
            cf_survivors = self.run_iteration(self.num_cycle, fa_seed, cf_parents)
            self.wrapup_cycle(**kwargs)
            cf_parents = cf_survivors

        # self.statistics
        final_population = cf_parents
        final_fitness = self.set_population_fitness(final_population, fa_seed)
        
        return final_fitness, self.stats

    def run_iteration(self, cycle_num: int, fa_seed: Cases, cf_parents: MutatedCases):
        self.curr_stats.update_base("num_cycle", cycle_num)

        cf_offspring = self.generate_offspring(cf_parents, fa_seed)
        self.curr_stats.update_base("num_offspring", cf_offspring.size)
        self.curr_stats.update_mutations('mut_num_o', cf_offspring.mutations)

        cf_offspring = self.set_population_fitness(cf_offspring, fa_seed)
        self.curr_stats.update_base("avg_offspring_fitness", cf_offspring.avg_viability)

        cf_survivors = self.pick_survivors(cf_offspring)
        self.curr_stats.update_base("num_survivors", cf_survivors.size)
        self.curr_stats.update_base("avg_survivors_fitness", cf_survivors.avg_viability)
        self.curr_stats.update_base("median_survivors_fitness", cf_survivors.median_viability)
        self.curr_stats.update_base("max_survivors_fitness", cf_survivors.max_viability)
        self.curr_stats.update_mutations('mut_num_s', cf_survivors.mutations)

        return cf_survivors

    def generate_offspring(self, cf_parents: MutatedCases, fc_seed: MutatedCases, **kwargs):
        if cf_parents is None:
            offspring = self._init_population(fc_seed)
            mutated = self._mutate_offspring(offspring, fc_seed)
            return mutated
        offspring = self._generate_population(cf_parents, fc_seed)
        mutated = self._mutate_offspring(offspring, fc_seed)
        return mutated

    @abstractmethod
    def _init_population(self, fc_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    @abstractmethod
    def set_population_fitness(self, cf_offspring: MutatedCases, fc_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    @abstractmethod
    def _generate_population(self, cf_parents: MutatedCases, fc_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    @abstractmethod
    def _recombine_parents(self, events, features, *args, **kwargs) -> MutatedCases:
        pass

    @abstractmethod
    def _mutate_offspring(self, cf_offspring: MutatedCases, fc_seed: MutatedCases, *args, **kwargs) -> MutatedCases:
        pass

    def pick_survivors(self, cf_offspring: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        mutations = cf_offspring.mutations
        ranking = np.argsort(fitness.viabs)
        selector = ranking[-self.num_survivors:].flatten()
        selected_fitness = fitness[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]
        selected_mutations = mutations[selector]
        selected = MutatedCases(selected_events, selected_features, selected_fitness).set_mutations(selected_mutations)
        return selected

    def wrapup_cycle(self, *args, **kwargs):
        self.num_cycle += 1
        self.cycle_pbar.update(1)
        self.statistics.attach(self.curr_stats)
        self.curr_stats = IterationStatistics()

    def is_cycle_end(self, *args, **kwargs) -> bool:
        return self.num_cycle >= self.max_iter

    @property
    def stats(self):
        return self.statistics.stats

    # @abstractmethod
    # def __call__(self, *args, **kwargs):
    #     pass