import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from thesis_generators.models.model_commons import BaseModelMixin

from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG_STOP = 1000


class EvolutionaryStrategy(BaseModelMixin, ABCMeta):
    evolutionary_counter: int = None
    max_iter: int = None

    def __init__(self, evaluator: ViabilityMeasure, max_iter: int = 1000, survival_thresh: int = 5, num_population: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fitness_function = evaluator
        self.max_iter = max_iter
        self.name = self.__class__.__name__
        self.num_survivors = survival_thresh
        self.num_population = num_population

    def __call__(self, factual_seeds):
        all_generated = []
        for fc_seed in self.__next_seed(factual_seeds):
            cf_offspring = self.generate_offspring(None, fc_seed)
            fitness_values = self.determine_fitness(cf_offspring)
            cf_survivors = self.pick_survivors(cf_offspring, fitness_values)
            cf_parents = cf_survivors

            while self.__is_termination(cf_survivors, fitness_values, self.evolutionary_counter, fc_seed):
                cf_offspring = self.generate_offspring(cf_parents, fc_seed)
                fitness_values = self.determine_fitness(cf_offspring, fc_seed)
                cf_survivors = self.pick_survivors(cf_offspring, fitness_values)
                cf_parents = cf_survivors
                self.__add_counter()

            final_population = cf_parents
            final_fitness = self.determine_fitness(final_population)
            all_generated.append((final_population, final_fitness))

        return all_generated

    def generate_offspring(self, cf_parents, factual_seed, **kwargs):
        if cf_parents is None:
            offspring = self._init_population(factual_seed)
            return offspring
        offspring = self._generate_population(cf_parents, factual_seed)
        mutated = self._mutate_offspring(offspring, factual_seed)
        return offspring

    def __next_seed(self, factual_seeds):
        fc_events, fc_features = factual_seeds
        max_len = len(fc_events)
        for i in enumerate(range(max_len)):
            yield fc_events[i], fc_features[i]

    @abstractmethod
    def initialize_population(self, *args, **kwargs):
        pass

    @abstractmethod
    def determine_fitness(self, cf_offspring, fc_seed, **kwargs):
        pass

    @abstractmethod
    def _generate_population(self, cf_offspring, fc_seed, **kwargs):
        pass

    @abstractmethod
    def _init_population(self, fc_seed, **kwargs):
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
        return selected

    def __add_counter(self, *args, **kwargs):
        self.evolutionary_counter += 1

    def __is_termination(self, *args, **kwargs):
        return self.evolutionary_counter >= self.max_iter
