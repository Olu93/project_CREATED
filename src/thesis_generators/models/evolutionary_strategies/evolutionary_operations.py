from __future__ import annotations
from abc import ABC, abstractmethod
from ast import Dict
import itertools
from numbers import Number
from typing import List, Tuple, Type, TYPE_CHECKING
# if TYPE_CHECKING:
#     from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy

from thesis_commons.functions import extract_padding_mask
from thesis_commons.random import random
from thesis_commons.modes import MutationMode
from thesis_commons.representations import BetterDict, Cases, Configuration, ConfigurationSet, MutatedCases, MutationRate
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.distributions import DataDistribution
import numpy as np
from numpy.typing import NDArray


#  https://cs.stackexchange.com/a/54835
class EvolutionaryOperatorInterface(Configuration):
    name: str = "NA"
    vocab_len: int = None
    sample_size: int = None
    fitness_function: ViabilityMeasure = None
    num_survivors: int = None

    def get_config(self):
        return BetterDict(super().get_config()).merge({'evo': {"num_survivors": self.num_survivors}})

    def set_fitness_function(self, fitness_function: ViabilityMeasure) -> EvolutionaryOperatorInterface:
        self.fitness_function = fitness_function
        return self

    def set_num_survivors(self, num_survivors: int) -> EvolutionaryOperatorInterface:
        self.num_survivors = num_survivors
        return self


class Initiator(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def init_population(self, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def get_config(self):
        return BetterDict(super().get_config()).merge({"initiator": {'type': type(self).__name__}})


class Selector(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def get_config(self):
        return BetterDict(super().get_config()).merge({"selector": {'type': type(self).__name__}})


class Crosser(EvolutionaryOperatorInterface, ABC):
    crossover_rate: Number = None

    @abstractmethod
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def set_crossover_rate(self, crossover_rate: float) -> EvolutionaryOperatorInterface:
        self.crossover_rate = crossover_rate
        return self

    def get_parent_ids(self, cf_ev, total) -> Tuple[NDArray, NDArray]:
        ids: NDArray = random.integers(0, len(cf_ev), size=(2, total))
        mother_ids: NDArray = ids[0]
        father_ids: NDArray = ids[1]
        return mother_ids, father_ids

    def get_config(self):
        return BetterDict(super().get_config()).merge({'crosser': {'type': type(self).__name__, 'crossover_rate': self.crossover_rate}})

    def birth_twins(self, mother_events, father_events, mother_features, father_features, gene_flips):
        child_events1, child_features1 = self.birth(mother_events, father_events, mother_features, father_features, gene_flips)
        child_events2, child_features2 = self.birth(mother_events, father_events, mother_features, father_features, ~gene_flips)
        child_events = np.vstack([child_events1, child_events2])
        child_features = np.vstack([child_features1, child_features2])
        return child_events, child_features

    def birth(self, mother_events, father_events, mother_features, father_features, gene_flips):
        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]
        return child_events, child_features


class OnePointCrosser(Crosser):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = self.sample_size
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = self.get_parent_ids(cf_ev, total)
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]

        positions = np.ones((total, mother_events.shape[1])) * np.arange(0, mother_events.shape[1])[None]
        cut_points = random.integers(0, mother_events.shape[1], size=total)[:, None]

        gene_flips = positions > cut_points
        child_events, child_features = self.birth_twins(mother_events, father_events, mother_features, father_features, gene_flips)

        return MutatedCases(child_events, child_features)


class TwoPointCrosser(Crosser):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = self.sample_size
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = self.get_parent_ids(cf_ev, total)
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]

        positions = np.ones((total, mother_events.shape[1])) * np.arange(0, mother_events.shape[1])[None]
        start_point = random.integers(0, mother_events.shape[1] - 1, size=total)[:, None]
        end_point = random.integers(start_point.flatten(), mother_events.shape[1])[:, None]

        gene_flips = (start_point < positions) & (positions <= end_point)

        child_events, child_features = self.birth_twins(mother_events, father_events, mother_features, father_features, gene_flips)

        return MutatedCases(child_events, child_features)


class UniformCrosser(Crosser):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = self.sample_size
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = self.get_parent_ids(cf_ev, total)
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]
        mask = extract_padding_mask(mother_events)
        gene_flips = random.random((total, mother_events.shape[1])) < self.crossover_rate
        gene_flips = gene_flips & mask
        child_events, child_features = self.birth_twins(mother_events, father_events, mother_features, father_features, gene_flips)
        return MutatedCases(child_events, child_features)


class Mutator(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def mutation(self, cf_offspring: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def set_mutation_rate(self, mutation_rate: MutationRate) -> Mutator:
        self.mutation_rate = mutation_rate
        return self

    def set_edit_rate(self, edit_rate: float) -> Mutator:
        self.edit_rate = edit_rate
        return self

    def get_config(self):
        return BetterDict(super().get_config()).merge({'mutator': {'type': type(self).__name__, **self.mutation_rate.get_config()}})


class Recombiner(EvolutionaryOperatorInterface, ABC):
    recombination_rate: Number = None

    @abstractmethod
    def recombination(self, cf_offspring: MutatedCases, cf_population: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def set_recombination_rate(self, recombination_rate: float) -> Recombiner:
        self.recombination_rate = recombination_rate
        return self

    def get_config(self):
        return BetterDict(super().get_config()).merge({'recombiner': {'type': type(self).__name__, 'recombination_rate': self.recombination_rate}})


class FittestIndividualRecombiner(Recombiner):
    def recombination(self, cf_mutated: MutatedCases, cf_population: MutatedCases, **kwargs) -> MutatedCases:
        cf_offspring = cf_mutated + cf_population
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        ranking = np.argsort(fitness.viabs, axis=0)
        selector = ranking[-self.num_survivors:].flatten()
        selected_fitness = fitness[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]
        selected = MutatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        return selected


class BestBreedRecombiner(Recombiner):
    def recombination(self, cf_offspring: MutatedCases, cf_population: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev_offspring, cf_ft_offspring, _, offspring_fitness = cf_offspring.all
        # mutations = cf_offspring.mutations
        selector = random.random(size=len(offspring_fitness.viabs)) < 0.5
        selected_fitness = offspring_fitness[selector]
        selected_events = cf_ev_offspring[selector]
        selected_features = cf_ft_offspring[selector]
        # selected_mutations = mutations[selector]
        selected_offspring = MutatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        selected = selected_offspring + cf_population
        return selected


class DefaultInitiator(Initiator):
    def init_population(self, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        fc_ev, fc_ft = fa_seed.cases
        random_events = random.integers(0, self.vocab_len, (self.sample_size, ) + fc_ev.shape[1:]).astype(float)
        random_features = random.standard_normal((self.sample_size, ) + fc_ft.shape[1:])
        return MutatedCases(random_events, random_features).evaluate_fitness(self.fitness_function, fa_seed)


class CaseBasedInitiator(Initiator):
    def init_population(self, fa_seed: MutatedCases, **kwargs):
        vault: Cases = self.vault
        all_cases = vault.sample(self.sample_size)
        events, features = all_cases.cases
        return MutatedCases(events, features).evaluate_fitness(self.fitness_function, fa_seed)

    def set_vault(self, vault: Cases) -> CaseBasedInitiator:
        self.vault = vault
        return self


class DataDistributionSampleInitiator(Initiator):
    def init_population(self, fa_seed: MutatedCases, **kwargs):
        dist: DataDistribution = self.data_distribution
        sampled_cases = dist.sample(self.sample_size)
        events, features = sampled_cases.cases
        return MutatedCases(events, features).evaluate_fitness(self.fitness_function, fa_seed)

    def set_data_distribution(self, data_distribution: DataDistribution) -> DataDistributionSampleInitiator:
        self.data_distribution = data_distribution
        return self


class RouletteWheelSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        normalized_viabs = viabs / viabs.sum()
        selection = random.choice(np.arange(len(cf_population)), size=self.num_survivors, p=normalized_viabs)
        cf_selection = cf_population[selection]
        return cf_selection


class TournamentSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        num_contenders = len(cf_population)
        # Seems Superior
        left_corner = random.choice(np.arange(0, num_contenders), size=self.sample_size)
        right_corner = random.choice(np.arange(0, num_contenders), size=self.sample_size)
        # Seems Inferior
        # left_corner = random.choice(np.arange(0, num_contenders), size=num_contenders, replace=False)
        # right_corner = random.choice(np.arange(0, num_contenders), size=num_contenders, replace=False)
        left_is_winner = viabs[left_corner] > viabs[right_corner]

        winners = np.ones_like(left_corner)
        winners[left_is_winner] = left_corner[left_is_winner]
        winners[~left_is_winner] = right_corner[~left_is_winner]

        cf_selection = cf_population[winners]
        return cf_selection


class ElitismSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        ranking = np.argsort(viabs, axis=0)
        selector = ranking[-self.num_survivors:]
        cf_selection = cf_population[selector]
        return cf_selection


class SingleDeleteMutator(Mutator):
    def mutation(self, cf_offspring: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        events, features = cf_offspring.cases
        # This corresponds to one Mutation per Case
        m_type = random.choice(MutationMode, size=(events.shape[0], 1), p=self.mutation_rate.probs)
        num_edits = int(events.shape[1] * self.edit_rate)
        positions = np.argsort(random.random(events.shape), axis=1)

        delete_mask, change_mask, insert_mask, swap_mask = self.create_mutation_masks(events, m_type, num_edits, positions)
        orig_ev = events.copy()
        orig_ft = features.copy()

        events, features = self.delete(events, features, delete_mask)
        events, features = self.substitute(events, features, change_mask)
        events, features = self.insert(events, features, insert_mask)
        events, features = self.transpose(events, features, swap_mask, random.random() < .5)

        mutations = m_type
        return MutatedCases(events, features).set_mutations(mutations).evaluate_fitness(self.fitness_function, fa_seed)

    def delete(self, events, features, delete_mask):
        events[delete_mask] = 0
        features[delete_mask] = 0
        return events, features

    def substitute(self, events, features, change_mask):
        events[change_mask] = random.integers(1, self.vocab_len, events.shape)[change_mask]
        features[change_mask] = random.standard_normal(features.shape)[change_mask]
        return events, features

    def insert(self, events, features, insert_mask):
        events[insert_mask] = random.integers(1, self.vocab_len, events.shape)[insert_mask]
        features[insert_mask] = random.standard_normal(features.shape)[insert_mask]
        return events, features

    def transpose(self, events, features, swap_mask, is_reverse=False):
        reversal = -1 if is_reverse else 1
        source_container = np.roll(events, reversal * -1, axis=1)
        tmp_container = np.ones_like(events) * np.nan
        tmp_container[swap_mask] = events[swap_mask]
        tmp_container = np.roll(tmp_container, reversal * 1, axis=1)
        backswap_mask = ~np.isnan(tmp_container)

        events[swap_mask] = source_container[swap_mask]
        events[backswap_mask] = tmp_container[backswap_mask]

        source_container = np.roll(features, reversal * -1, axis=1)
        tmp_container = np.ones_like(features) * np.nan
        tmp_container[swap_mask] = features[swap_mask]
        tmp_container = np.roll(tmp_container, reversal * 1, axis=1)

        features[swap_mask] = source_container[swap_mask]
        features[backswap_mask] = tmp_container[backswap_mask]
        return events, features

    def create_mutation_masks(self, events, m_type, num_edits, positions):
        delete_mask = (m_type == MutationMode.DELETE) & (events != 0) & (positions < 1)
        change_mask = (m_type == MutationMode.CHANGE) & (events != 0) & (positions <= num_edits)
        insert_mask = (m_type == MutationMode.INSERT) & (events == 0) & (positions <= num_edits)
        transp_mask = (m_type == MutationMode.TRANSP) & (positions <= num_edits)

        return delete_mask, change_mask, insert_mask, transp_mask


# TODO Add one that is randomly selecting from data dist
class MultiDeleteMutator(SingleDeleteMutator):
    def create_mutation_masks(self, events, m_type, num_edits, positions):
        delete_mask = (m_type == MutationMode.DELETE) & (events != 0) & (positions <= num_edits)
        change_mask = (m_type == MutationMode.CHANGE) & (events != 0) & (positions <= num_edits)
        insert_mask = (m_type == MutationMode.INSERT) & (events == 0) & (positions <= num_edits)
        transp_mask = (m_type == MutationMode.TRANSP) & (positions <= num_edits)
        return delete_mask, change_mask, insert_mask, transp_mask


class EvoConfigurator(ConfigurationSet):
    def __init__(self, initiator: Initiator, selector: Selector, crosser: Crosser, mutator: Mutator, recombiner: Recombiner):
        self.initiator = initiator
        self.selector = selector
        self.crosser = crosser
        self.mutator = mutator
        self.recombiner = recombiner
        self._list: List[EvolutionaryOperatorInterface] = [initiator, selector, crosser, mutator, recombiner]

    def set_fitness_function(self, evaluator: ViabilityMeasure) -> EvoConfigurator:
        for operator in self._list:
            self = operator.set_fitness_function(evaluator)
        return self

    def set_sample_size(self, sample_size: int) -> EvoConfigurator:
        for operator in self._list:
            self = operator.set_sample_size(sample_size)
        return self

    def set_num_survivors(self, num_survivors: int) -> EvoConfigurator:
        for operator in self._list:
            self = operator.set_num_survivors(num_survivors)
        return self

    def set_vocab_len(self, vocab_len: int) -> EvoConfigurator:
        for operator in self._list:
            self = operator.set_vocab_len(vocab_len)
        return self

    def __iter__(self) -> EvolutionaryOperatorInterface:
        for operator in self._list:
            yield operator

    def __repr__(self):
        return "_".join([type(op).__name__ for op in self._list])

    @staticmethod
    def registry(evaluator: ViabilityMeasure,
                 initiators: List[Initiator] = None,
                 selectors: List[Selector] = None,
                 crossers: List[Crosser] = None,
                 mutators: List[Mutator] = None,
                 recombiners: List[Recombiner] = None,
                 **kwargs):
        crossover_rate = kwargs.get('crossover_rate', 0.5)
        edit_rate = kwargs.get('edit_rate', 0.1)
        mutation_rate = kwargs.get('mutation_rate', MutationRate())
        recombination_rate = kwargs.get('recombination_rate', 0.5)
        initiators = initiators or [
            DefaultInitiator(),
            CaseBasedInitiator().set_vault(evaluator._training_data),
            DataDistributionSampleInitiator().set_data_distribution(evaluator.measures.dllh.data_distribution),
        ]
        selectors = selectors or [
            RouletteWheelSelector(),
            TournamentSelector(),
            ElitismSelector(),
        ]
        crossers = crossers or [
            OnePointCrosser(),
            TwoPointCrosser(),
            UniformCrosser().set_crossover_rate(crossover_rate),
        ]
        mutators = mutators or [
            # SingleDeleteMutator().set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
            MultiDeleteMutator().set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
        ]
        recombiners = recombiners or [
            FittestIndividualRecombiner(),
            BestBreedRecombiner(),
        ]

        combos = itertools.product(initiators, selectors, crossers, mutators, recombiners)
        result = [EvoConfigurator(*cnf) for cnf in combos]
        return result
