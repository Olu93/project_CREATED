from __future__ import annotations
from abc import ABC, abstractmethod
import io
from tokenize import Number
from typing import Any, Counter, Dict, List, Sequence, Tuple, Type, Union
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from thesis_commons.constants import PATH_RESULTS_MODELS_SPECIFIC
from thesis_commons.functions import extract_padding_mask

from thesis_commons.model_commons import BaseModelMixin
from thesis_commons.modes import MutationMode
from thesis_commons.representations import BetterDict, Cases, EvaluatedCases, EvaluatedCases, MutationRate
from thesis_commons.statistics import StatInstance, StatIteration, StatRow, attach_descriptive_stats
from thesis_generators.models.evolutionary_strategies.evolutionary_operations import Crosser, EvoConfigurator, Initiator, Mutator, Recombiner, Selector
from thesis_viability.viability.viability_function import ViabilityMeasure

DEBUG_STOP = 1000
DEBUG_VERBOSE = True
# https://cs.stackexchange.com/a/54835


# TODO: Rename num_population to sample size
# TODO: Rename survival_thresh to num_survivors

class EvolutionaryStrategy(BaseModelMixin):
    def __init__(self, evaluator: ViabilityMeasure, operators: EvoConfigurator, max_iter: int = 1000, survival_thresh: int = 25, num_population: int = 100, **kwargs) -> None:
        super(EvolutionaryStrategy, self).__init__(**kwargs)
        self.fitness_function = evaluator
        self.operators = operators
        self.operators.set_fitness_function(evaluator)
        self.operators.set_vocab_len(self.vocab_len)
        self.operators.set_num_survivors(survival_thresh)
        self.operators.set_sample_size(num_population)

        self.max_iter: int = max_iter
        # self.name: str = "Evo_" + repr(operators)
        self.num_survivors: int = survival_thresh
        self.num_population: int = num_population
        self.num_cycle: int = 0
        self._iteration_statistics: StatIteration = None
        self._curr_stats: StatRow = None
        self.cycle_pbar: tqdm = None
        self.is_saved: bool = False
        # self._stats: Sequence[IterationStatistics] = []

    def predict(self, fa_case: Cases, **kwargs) -> Tuple[EvaluatedCases, StatIteration]:
        fa_seed = Cases(*fa_case.all)
        self.instance_stats = StatInstance()
        self._iteration_statistics = StatIteration()
        cf_parents: EvaluatedCases = self.operators.initiator.init_population(fa_seed, **kwargs)
        cf_survivors: EvaluatedCases = cf_parents
        self.num_cycle = 0
        self.cycle_pbar = tqdm(total=self.max_iter, desc="Evo Cycle") if DEBUG_VERBOSE else None

        while not self.is_cycle_end(cf_survivors, self.num_cycle, fa_seed):
            self._curr_stats = StatRow()
            cf_survivors = self.run_iteration(self.num_cycle, fa_seed, cf_parents)
            self.wrapup_cycle(**kwargs)
            cf_parents = cf_survivors

        # self.statistics
        final_population = cf_parents
        # final_fitness = self.set_population_fitness(final_population, fa_seed)
        # for
        # self.is_saved:bool = self.save_statistics()
        # if self.is_saved:
        #     print("Successfully saved stats!")
        self._iteration_statistics.attach("initiator", type(self.operators.initiator).__name__)
        self._iteration_statistics.attach("selector", type(self.operators.selector).__name__)
        self._iteration_statistics.attach("crosser", type(self.operators.crosser).__name__)
        self._iteration_statistics.attach("mutator", type(self.operators.mutator).__name__)
        # self._iteration_statistics.attach("recombiner", type(self.operators.recombiner).__name__)
        # self.instance_stats.attach("config", self.operators.get_config())
        return final_population, self.instance_stats

    def run_iteration(self, cycle_num: int, fa_seed: Cases, cf_population: EvaluatedCases, **kwargs):
        self._curr_stats.attach("num_cycle", cycle_num)
        # if cycle_num ==3:
        #     print("EVO STOP")
        cf_offspring = self.operators.crosser.crossover(cf_population, fa_seed, **kwargs)
        cf_mutated = self.operators.mutator.mutation(cf_offspring, fa_seed, **kwargs)
        cf_candidates = cf_mutated + cf_population
        cf_candidates = cf_candidates.evaluate_viability(self.fitness_function, fa_seed)
        cf_survivors = self.operators.selector.selection(cf_candidates, fa_seed, **kwargs)
        # cf_survivors = self.operators.recombiner.recombination(cf_mutated, cf_population, **kwargs)

        
        self._curr_stats.attach("n_population", cf_population.size)
        self._curr_stats.attach("n_offspring", cf_offspring.size)
        self._curr_stats.attach("n_mutated", cf_mutated.size)
        self._curr_stats.attach("n_survivors", cf_survivors.size)
        self._curr_stats.attach("n_candidates", cf_candidates.size)                                                                                                                            
        self._curr_stats.attach('mutsum', cf_mutated, EvolutionaryStrategy.count_mutations)
        
        self._iteration_statistics = attach_descriptive_stats(self._iteration_statistics, cf_survivors)

        return cf_survivors



    def set_population_fitness(self, cf_offspring: EvaluatedCases, fc_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        fitness = self.fitness_function(fc_seed, cf_offspring)
        return cf_offspring.set_viability(fitness)

    def wrapup_cycle(self, *args, **kwargs):
        self.num_cycle += 1
        if DEBUG_VERBOSE:
            self.cycle_pbar.update(1)
            sys.stdout.flush()
        self._iteration_statistics.append(self._curr_stats)
        self._iteration_statistics = StatIteration()
        self.instance_stats.append(self._iteration_statistics)

    def is_cycle_end(self, *args, **kwargs) -> bool:
        return self.num_cycle >= self.max_iter

    def get_config(self) -> Dict:
        return BetterDict(super().get_config()).merge({"max_iter": self.max_iter, "num_survivors": self.num_survivors}).merge(self.operators.get_config())

    @property
    def stats(self):
        return self._iteration_statistics.data

    @staticmethod
    def count_mutations(cases: EvaluatedCases):
        x = cases.mutations.flatten()
        cnt = Counter(list(x))
        sum_of_counts = sum(list(cnt.values()))
        result = {mtype._name_: cnt.get(mtype, 0)/sum_of_counts for mtype in MutationMode}
        return result

    # def build(self, initiator: Initiator, selector: Selector, crosser: Crosser, mutator: Mutator, recombiner: Recombiner) -> EvolutionaryStrategy:
    #     return self.set_initializer(initiator).set_selector(selector).set_crosser(crosser).set_mutator(mutator).set_recombiner(recombiner)
