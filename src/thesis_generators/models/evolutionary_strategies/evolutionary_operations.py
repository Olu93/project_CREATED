from __future__ import annotations
from abc import ABC, abstractmethod
from ast import Dict
import itertools
from typing import List, Tuple, Type, TYPE_CHECKING
# if TYPE_CHECKING:
#     from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy

from thesis_commons.functions import extract_padding_mask, merge_dicts
from thesis_commons.random import random
from thesis_commons.modes import MutationMode
from thesis_commons.representations import Cases, Configuration, ConfigurationSet, MutatedCases, MutationRate
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.distributions import DataDistribution
import numpy as np
from numpy.typing import NDArray


class EvolutionaryOperatorInterface(Configuration):
    name: str = "NA"
    vocab_len: int = None
    sample_size: int = None
    fitness_function: ViabilityMeasure = None
    num_survivors: int = None

    def get_config(self):
        return merge_dicts(super().get_config(), {'evo': {"num_survivors": self.num_survivors}})

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
        return merge_dicts(super().get_config(), {"initiator": {'type': type(self).__name__}})


class Selector(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def get_config(self):
        return merge_dicts(super().get_config(), {"selector": {'type': type(self).__name__}})


class Crosser(EvolutionaryOperatorInterface, ABC):
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
        return merge_dicts(super().get_config(), {'crosser': {'type': type(self).__name__, 'crossover_rate': self.crossover_rate}})


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
        return merge_dicts(super().get_config(), {'mutator': {'type': type(self).__name__, **self.mutation_rate.get_config()}})


class Recombiner(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def recombination(self, cf_offspring: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        pass

    def set_recombination_rate(self, recombination_rate: float) -> Recombiner:
        self.recombination_rate = recombination_rate
        return self

    def get_config(self):
        return merge_dicts(super().get_config(), {'recombiner': {'type': type(self).__name__, 'recombination_rate': self.recombination_rate}})


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
        selection = random.choice(np.arange(len(cf_population)), size=100, p=normalized_viabs)
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


class CutPointCrosser(Crosser):
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
        take_pre = random.random(size=total)[:, None] > self.inheritance_swapping_rate  #TODO: Doesn't cut all
        gene_flips = positions > cut_points
        gene_flips[take_pre.flatten()] = ~gene_flips[take_pre.flatten()]

        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]

        return MutatedCases(child_events, child_features)

    def set_inheritance_swapping_rate(self, cut_position_rate: float) -> CutPointCrosser:
        self.inheritance_swapping_rate = cut_position_rate
        return self


class NPointCrosser(Crosser):
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
        gene_flips = random.random((total, mother_events.shape[1])) > self.crossover_rate
        gene_flips = gene_flips & mask
        child_events = np.copy(mother_events)
        child_events[gene_flips] = father_events[gene_flips]
        child_features = np.copy(mother_features)
        child_features[gene_flips] = father_features[gene_flips]

        return MutatedCases(child_events, child_features)


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
        events, features = self.transpose(events, features, swap_mask)

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

    def transpose(self, events, features, swap_mask):
        source_container = np.roll(events, -1, axis=1)
        tmp_container = np.ones_like(events) * np.nan
        tmp_container[swap_mask] = events[swap_mask]
        tmp_container = np.roll(tmp_container, 1, axis=1)
        backswap_mask = ~np.isnan(tmp_container)

        events[swap_mask] = source_container[swap_mask]
        events[backswap_mask] = tmp_container[backswap_mask]

        source_container = np.roll(features, -1, axis=1)
        tmp_container = np.ones_like(features) * np.nan
        tmp_container[swap_mask] = features[swap_mask]
        tmp_container = np.roll(tmp_container, 1, axis=1)

        features[swap_mask] = source_container[swap_mask]
        features[backswap_mask] = tmp_container[backswap_mask]
        return events, features

    def create_mutation_masks(self, events, m_type, num_edits, positions):
        delete_mask = (m_type == MutationMode.DELETE) & (events != 0) & (positions < 1)
        change_mask = (m_type == MutationMode.CHANGE) & (events != 0) & (positions <= num_edits)
        insert_mask = (m_type == MutationMode.INSERT) & (events == 0) & (positions <= num_edits)
        transp_mask = (m_type == MutationMode.TRANSP) & (positions <= num_edits)

        return delete_mask, change_mask, insert_mask, transp_mask


class MultiDeleteMutator(SingleDeleteMutator):
    def create_mutation_masks(self, events, m_type, num_edits, positions):
        delete_mask = (m_type == MutationMode.DELETE) & (events != 0) & (positions <= num_edits)
        change_mask = (m_type == MutationMode.CHANGE) & (events != 0) & (positions <= num_edits)
        insert_mask = (m_type == MutationMode.INSERT) & (events == 0) & (positions <= num_edits)
        transp_mask = (m_type == MutationMode.TRANSP) & (positions <= num_edits)
        return delete_mask, change_mask, insert_mask, transp_mask


class DefaultRecombiner(Recombiner):
    def recombination(self, cf_offspring: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        # mutations = cf_offspring.mutations
        ranking = np.argsort(fitness.viabs, axis=0)
        selector = ranking[-self.num_survivors:].flatten()
        selected_fitness = fitness[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]
        # selected_mutations = mutations[selector]
        selected = MutatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        return selected


class EvoConfig(ConfigurationSet):
    def __init__(self, initiator: Initiator, selector: Selector, crosser: Crosser, mutator: Mutator, recombiner: Recombiner):
        self.initiator = initiator
        self.selector = selector
        self.crosser = crosser
        self.mutator = mutator
        self.recombiner = recombiner
        self._list: List[EvolutionaryOperatorInterface] = [self.initiator, self.selector, self.crosser, self.mutator, self.recombiner]

    @property
    def list_of_operators(self):
        return self._list

    def set_fitness_function(self, evaluator: ViabilityMeasure) -> EvoConfig:
        for operator in self.list_of_operators:
            self = operator.set_fitness_function(evaluator)
        return self

    def set_sample_size(self, sample_size: int) -> EvoConfig:
        for operator in self.list_of_operators:
            self = operator.set_sample_size(sample_size)
        return self

    def set_num_survivors(self, num_survivors: int) -> EvoConfig:
        for operator in self.list_of_operators:
            self = operator.set_num_survivors(num_survivors)
        return self

    def set_vocab_len(self, vocab_len: int) -> EvoConfig:
        for operator in self.list_of_operators:
            self = operator.set_vocab_len(vocab_len)
        return self

    def __iter__(self) -> EvolutionaryOperatorInterface:
        for operator in self.list_of_operators:
            yield operator

    def __repr__(self):
        return "_".join([type(op).__name__ for op in self.list_of_operators])


class EvoConfigurator():
    def __init__(self,
                 initiators: List[Initiator] = None,
                 selectors: List[Selector] = None,
                 crossers: List[Crosser] = None,
                 mutators: List[Mutator] = None,
                 recombiners: List[Recombiner] = None):
        self.initiators = initiators
        self.selectors = selectors
        self.crossers = crossers
        self.mutators = mutators
        self.recombiners = recombiners

    @staticmethod
    def registry(fitness_func: ViabilityMeasure, **kwargs) -> EvoConfig:
        edit_rate = kwargs.get('edit_rate', 0.1)
        crossover_rate = kwargs.get('crossover_rate', 0.1)
        mutation_rate = kwargs.get('mutation_rate', MutationRate())
        recombination_rate = kwargs.get('recombination_rate', 0.5)
        inheritance_swapping_rate = kwargs.get('inheritance_swapping_rate', 0.5)
        initiators = [
            DefaultInitiator(),
            CaseBasedInitiator().set_vault(fitness_func._training_data),
            # DataDistributionSampleInitiator().set_data_distribution(fitness_func.datalikelihood_computer.data_distribution),
        ]

        selectors = [
            RouletteWheelSelector(),
            TournamentSelector(),
            ElitismSelector(),
        ]

        crossers = [
            CutPointCrosser().set_inheritance_swapping_rate(inheritance_swapping_rate).set_crossover_rate(crossover_rate),
            NPointCrosser().set_crossover_rate(crossover_rate),
        ]

        mutators = [
            SingleDeleteMutator().set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
            MultiDeleteMutator().set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
        ]

        recombiners = [
            DefaultRecombiner().set_recombination_rate(recombination_rate),
        ]

        config = EvoConfigurator(initiators=initiators, selectors=selectors, crossers=crossers, mutators=mutators, recombiners=recombiners)
        return config

    @staticmethod
    def combinations(selection: EvoConfigurator = None, evaluator: ViabilityMeasure = None, **kwargs) -> List[EvoConfig]:
        selection = selection if selection is not None else EvoConfigurator.registry(evaluator, **kwargs)
        combos = itertools.product(selection.initiators, selection.selectors, selection.crossers, selection.mutators, selection.recombiners)
        result = [EvoConfig(*cnf) for cnf in combos]
        return result
