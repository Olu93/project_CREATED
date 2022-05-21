from tokenize import Number
from typing import List, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from thesis_generators.models.model_commons import BaseModelMixin
from tqdm import tqdm
from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG_STOP = 1000


class IterationStatistics():
    def __init__(self, instance_num: int) -> None:
        self.base_store = {}
        self.complex_store = {}
        self.instance_num: int = instance_num

    # num_generation, num_population, num_survivors, fitness_values
    def base_update(self, key: str, val: Number):
        self.base_store[key] = val


class GlobalStatistics():
    def __init__(self) -> None:
        self.store = []

    def attach(self, iteration_stats: IterationStatistics):
        self.store.append(iteration_stats)

    def compute(self, selection: List[int] = None):

        if selection is None:
            base = [stats.base_store for stats in self.store if stats.instance_num in selection]
            self.stats = pd.DataFrame(base)
            return self
        base = [stats.base_store for stats in self.store if stats.instance_num in selection]
        self.stats = pd.DataFrame(base)
        return self

    def stats(self, ) -> pd.DataFrame:
        return self.stats


class Population():
    def __init__(self, events: np.ndarray, features: np.ndarray):
        self.events = events
        self.features = features
        self.num_cases, self.max_len, self.num_features = features.shape
        self.fitness = None
        self.survives = None


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

    def __call__(self, factual_seeds, labels):
        all_generated = []
        for instance_num, (fc_seed, fc_outcome) in enumerate(self.__next_seed(factual_seeds, labels)):
            self.curr_stats = IterationStatistics(instance_num)
            self.statistics.attach(self.curr_stats)
            cycle_num = 0
            cf_parents = None
            cf_survivors, fitness_values = self.run_iteration(instance_num, cycle_num, fc_seed, cf_parents)

            while self.__is_termination(cf_survivors, fitness_values, self.evolutionary_counter, fc_seed):
                cf_survivors, fitness_values = self.run_iteration(instance_num, cycle_num, fc_seed, cf_parents)
                cf_parents = cf_survivors

            final_population = cf_parents
            final_fitness = self.determine_fitness(final_population, fc_seed)
            all_generated.append((final_population, final_fitness))

        return all_generated

    def run_iteration(self, instance_num, cycle_num, fc_seed, cf_parents):
        self.curr_stats.base_update("cycle", cycle_num)

        cf_offspring = self.generate_offspring(cf_parents, fc_seed)
        self.curr_stats.base_update("offspring_num", len(cf_offspring[0]))

        fitness_values = self.determine_fitness(cf_offspring, fc_seed)[0]
        self.curr_stats.base_update("offspring_fitness_avg", fitness_values.mean())

        cf_survivors, survivor_fitness = self.pick_survivors(cf_offspring, fitness_values)
        self.curr_stats.base_update("survivors_num", len(cf_survivors[0]))
        self.curr_stats.base_update("survivors_fitness_avg", survivor_fitness.mean())

        self.finish_evo_cycle(instance_num)
        return cf_survivors, survivor_fitness

    def generate_offspring(self, cf_parents, fc_seed, **kwargs):
        if cf_parents is None:
            offspring = self._init_population(fc_seed)
            return offspring
        offspring = self._generate_population(cf_parents, fc_seed)
        mutated = self._mutate_offspring(offspring, fc_seed)
        return mutated

    def __next_seed(self, factual_seeds, labels):
        fc_events, fc_features = factual_seeds
        max_len = len(fc_events)
        for i in range(max_len):
            yield (fc_events[i][None, ...], fc_features[i][None, ...]), labels[i]

    @abstractmethod
    def _init_population(self, fc_seed, **kwargs):
        pass

    @abstractmethod
    def determine_fitness(self, cf_offspring, fc_seed, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def _generate_population(self, cf_offspring, fc_seed, **kwargs):
        pass

    @abstractmethod
    def _recombine_parents(self, events, features, *args, **kwargs):
        pass

    @abstractmethod
    def _mutate_offspring(self, cf_offspring, factual_seed, *args, **kwargs):
        pass

    def pick_survivors(self, cf_offspring, fitness_values, **kwargs):
        cf_ev, cf_ft = cf_offspring
        ranking = np.argsort(fitness_values)
        max_rank = np.max(ranking)
        selector = ranking > (max_rank - self.num_survivors)
        selected = cf_ev[selector], cf_ft[selector]
        return selected, fitness_values[selector]

    def finish_evo_cycle(self, instance_num, *args, **kwargs):
        self.curr_stats = IterationStatistics(instance_num)
        self.statistics.attach(self.curr_stats)
        self.evolutionary_counter += 1

    def __is_termination(self, *args, **kwargs):
        return self.evolutionary_counter >= self.max_iter

    # @abstractmethod
    # def __call__(self, *args, **kwargs):
    #     pass