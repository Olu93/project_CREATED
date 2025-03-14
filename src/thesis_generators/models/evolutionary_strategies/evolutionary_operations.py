from __future__ import annotations
from abc import ABC, abstractmethod
from ast import Dict
import itertools
from numbers import Number
from typing import List, Tuple, Type, TYPE_CHECKING
# if TYPE_CHECKING:
#     from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy

from thesis_commons.functions import extract_padding_end_indices, extract_padding_mask
from thesis_commons.random import random
from thesis_commons.modes import MutationMode
from thesis_commons.representations import BetterDict, Cases, Configuration, ConfigurationSet, EvaluatedCases, MutationRate, Viabilities
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.distributions import DataDistribution
import numpy as np
import pandas as pd
from collections import Counter

# from numpy.typing import np.ndarray


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


class HierarchicalMixin:
    def _compute(self, cf_cases: EvaluatedCases):

        tmp_result: EvaluatedCases = None
        tmp_cases = cf_cases
        tmp_res = tmp_cases.viabilities

        round_1 = np.stack([tmp_res.get(Viabilities.Measures.DATA_LLH).flatten(), tmp_res.get(Viabilities.Measures.OUTPUT_LLH).flatten()]).T
        mask = self.is_pareto_efficient(round_1)

        nondominated = np.where(mask)[0]
        dominated = np.where(~mask)[0]

        tmp_result = EvaluatedCases.from_cases(tmp_cases[nondominated])
        tmp_cases = tmp_cases[dominated]
        tmp_res = tmp_cases.viabilities

        round_2 = np.stack([tmp_res.get(Viabilities.Measures.OUTPUT_LLH).flatten(), tmp_res.get(Viabilities.Measures.SPARCITY).flatten()]).T
        mask = self.is_pareto_efficient(round_2)
        # nondominated_2 = np.where(tmp_cases[mask])[0]

        nondominated = np.where(mask)[0]
        dominated = np.where(~mask)[0]

        tmp_result = tmp_result + tmp_cases[nondominated]
        tmp_cases = tmp_cases[dominated]
        tmp_res = tmp_cases.viabilities

        round_3 = np.stack([tmp_res.get(Viabilities.Measures.SPARCITY).flatten(), tmp_res.get(Viabilities.Measures.SIMILARITY).flatten()]).T
        mask = self.is_pareto_efficient(round_3)

        nondominated = np.where(mask)[0]
        dominated = np.where(~mask)[0]

        result = tmp_result + tmp_cases[nondominated]
        remaining = tmp_cases[dominated]
        return result, remaining

    def order(self, cf_cases):
        res, rem = self._compute(cf_cases)
        result = res + rem
        return result

    def filter(self, cf_cases):
        res, rem = self._compute(cf_cases)
        result = res
        return result

    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


class ParetoMixin(HierarchicalMixin):
    def _compute(self, cf_cases: EvaluatedCases):
        tmp_result: EvaluatedCases = None
        tmp_cases = cf_cases
        tmp_res = tmp_cases.viabilities

        round_1 = np.stack([
            tmp_res.get(Viabilities.Measures.DATA_LLH).flatten(),
            tmp_res.get(Viabilities.Measures.OUTPUT_LLH).flatten(),
            tmp_res.get(Viabilities.Measures.SIMILARITY).flatten(),
            tmp_res.get(Viabilities.Measures.SPARCITY).flatten(),
        ]).T
        mask = self.is_pareto_efficient(round_1)
        nondominated = np.where(mask)[0]
        dominated = np.where(~mask)[0]

        result = tmp_cases[nondominated]
        remaining = tmp_cases[dominated]
        return result, remaining


class Initiator(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def init_population(self, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        pass

    def get_config(self):
        return BetterDict(super().get_config()).merge({"initiator": {'type': type(self).__name__}})


class RandomInitiator(Initiator):
    def init_population(self, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        fc_ev, fc_ft = fa_seed.cases
        ssize = self.sample_size
        random_events = random.integers(0, self.vocab_len, (ssize, ) + fc_ev.shape[1:]).astype(float)
        random_features = random.standard_normal((ssize, ) + fc_ft.shape[1:])
        return EvaluatedCases(random_events, random_features).evaluate_viability(self.fitness_function, fa_seed)


class FactualInitiator(Initiator):
    def init_population(self, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        fc_ev, fc_ft = fa_seed.cases
        # fc_ev = np.repeat(fc_ev, self.sample_size, axis=0)
        # fc_ft = np.repeat(fc_ft, self.sample_size, axis=0)
        return EvaluatedCases(fc_ev, fc_ft).evaluate_viability(self.fitness_function, fa_seed)


class CaseBasedInitiator(Initiator):
    def init_population(self, fa_seed: EvaluatedCases, **kwargs):
        vault: Cases = self.vault.original_data
        ssize = self.sample_size

        all_cases = vault.sample(ssize, replace=True)
        events, features = all_cases.cases
        return EvaluatedCases(events, features).evaluate_viability(self.fitness_function, fa_seed)

    def set_vault(self, vault: DataDistribution) -> CaseBasedInitiator:
        self.vault = vault
        return self


class SamplingBasedInitiator(Initiator):
    def init_population(self, fa_seed: EvaluatedCases, **kwargs):
        ssize = self.sample_size

        dist: DataDistribution = self.data_distribution

        sampled_cases = dist.sample(ssize)
        events, features = sampled_cases.cases
        return EvaluatedCases(events, features).evaluate_viability(self.fitness_function, fa_seed)

    def set_data_distribution(self, data_distribution: DataDistribution) -> SamplingBasedInitiator:
        self.data_distribution = data_distribution
        return self


class Selector(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def selection(self, cf_population: EvaluatedCases, cf_mutated: EvaluatedCases, **kwargs) -> EvaluatedCases:
        pass

    def get_config(self):
        return BetterDict(super().get_config()).merge({"selector": {'type': type(self).__name__}})


class RouletteWheelSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_candidates = cf_population

        evs, fts, llhs, fitness = cf_candidates.all
        ssize = self.sample_size

        viabs = fitness.viabs.flatten()
        normalized_viabs = viabs / viabs.sum()
        selection = random.choice(np.arange(len(cf_candidates)), size=ssize, p=normalized_viabs)
        cf_selection = cf_candidates[selection]
        return cf_selection


class TournamentSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_candidates = cf_population
        evs, fts, llhs, fitness = cf_candidates.all
        ssize = self.sample_size

        viabs = fitness.viabs.flatten()
        num_contenders = len(cf_candidates)
        # Seems Superior
        # left_corner = random.choice(np.arange(0, num_contenders), size=num_survivors)
        # right_corner = random.choice(np.arange(0, num_contenders), size=num_survivors)
        # Seems Inferior
        left_corner = random.choice(np.arange(0, num_contenders), size=ssize, replace=True)
        right_corner = random.choice(np.arange(0, num_contenders), size=ssize, replace=True)
        left_is_winner = viabs[left_corner] > viabs[right_corner]

        probs = np.ones((ssize, 2)) * np.array([0.25, 0.75])
        choices = np.ones((ssize, 2)) * np.array([2, 1])
        choices[~left_is_winner] = choices[~left_is_winner, ::-1]

        chosen = random.choice(choices.T, p=[0.25, 0.75])
        chosen_idx1 = np.where(chosen == 1)
        chosen_idx2 = np.where(chosen == 2)
        winner1 = left_corner[chosen_idx1]
        winner2 = right_corner[chosen_idx2]
        selector = np.concatenate([winner1, winner2])

        cf_selection = cf_population[selector]
        return cf_selection


class ElitismSelector(Selector):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_candidates = cf_population
        evs, fts, llhs, fitness = cf_candidates.all
        ssize = self.sample_size

        viabs = fitness.viabs.flatten()
        ranking = np.argsort(viabs, axis=0)
        selector = ranking[-ssize:]
        cf_selection = cf_candidates[selector]
        return cf_selection


class TopKsSelector(HierarchicalMixin, Selector):
    def selection(self, cf_population: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_candidates = cf_population
        cf_ev, cf_ft, _, fitness = cf_candidates.all
        ssize = self.sample_size
        cf_selection = cf_candidates[:ssize]

        return cf_selection


class UniformSampleSelector(HierarchicalMixin, Selector):
    def selection(self, cf_population: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_candidates = cf_population
        cf_ev, cf_ft, _, fitness = cf_candidates.all
        ssize = self.sample_size
        cf_selection = cf_candidates.sample(ssize, replace=True)
        return cf_selection


class Crosser(EvolutionaryOperatorInterface, ABC):
    crossover_rate: Number = None  # TODO: This is treated as a class attribute. Change to property

    @abstractmethod
    def crossover(self, cf_parents: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        pass

    def set_crossover_rate(self, crossover_rate: float) -> EvolutionaryOperatorInterface:
        self.crossover_rate = crossover_rate
        return self

    def get_parent_ids(self, cf_ev, total) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.arange(0, total)
        mother_ids: np.ndarray = random.permutation(ids)
        father_ids: np.ndarray = random.permutation(ids)
        return mother_ids, father_ids

    def get_config(self):
        return BetterDict(super().get_config()).merge(
            {'crosser': {
                'type': type(self).__name__ + (str(int((self.crossover_rate % 1) * 10)) if self.crossover_rate else ""),
                'crossover_rate': self.crossover_rate
            }})

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
    def crossover(self, cf_parents: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = len(cf_ev)
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = self.get_parent_ids(cf_ev, total)
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]

        positions = np.ones((total, mother_events.shape[1])) * np.arange(0, mother_events.shape[1])[None]
        cut_points = random.integers(0, mother_events.shape[1], size=total)[:, None]

        gene_flips = positions > cut_points
        child_events, child_features = self.birth_twins(mother_events, father_events, mother_features, father_features, gene_flips)

        return EvaluatedCases(child_events, child_features)


class TwoPointCrosser(Crosser):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = len(cf_ev)
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

        return EvaluatedCases(child_events, child_features)


class UniformCrosser(Crosser):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = len(cf_ev)
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = self.get_parent_ids(cf_ev, total)
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]
        mask = extract_padding_mask(mother_events)
        gene_flips = random.random((total, mother_events.shape[1])) < self.crossover_rate
        gene_flips = gene_flips & mask
        child_events, child_features = self.birth_twins(mother_events, father_events, mother_features, father_features, gene_flips)
        return EvaluatedCases(child_events, child_features)


class Mutator(EvolutionaryOperatorInterface, ABC):
    @abstractmethod
    def mutation(self, cf_offspring: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        pass

    def set_mutation_rate(self, mutation_rate: MutationRate) -> Mutator:
        self.mutation_rate = mutation_rate
        return self

    def set_edit_rate(self, edit_rate: float) -> Mutator:
        self.edit_rate = edit_rate
        return self

    def get_config(self):
        return BetterDict(super().get_config()).merge({'mutator': {'type': type(self).__name__, 'edit_rate': self.edit_rate, **self.mutation_rate.get_config()}})

    @abstractmethod
    def create_delete_mask(self, events, m_type, num_edits, positions) -> np.ndarray:
        pass

    @abstractmethod
    def create_insert_mask(self, events, m_type, num_edits, positions) -> np.ndarray:
        pass

    @abstractmethod
    def create_change_mask(self, events, m_type, num_edits, positions) -> np.ndarray:
        pass

    # @abstractmethod
    # def create_transp_mask(self, events, m_type, num_edits, positions) -> np.ndarray:
    #     pass


class RandomMutator(Mutator):
    def mutation(self, cf_offspring: EvaluatedCases, fa_seed: EvaluatedCases, **kwargs) -> EvaluatedCases:
        events, features = cf_offspring.cases
        events, features = events.copy(), features.copy()
        orig_ev = events.copy()
        orig_ft = features.copy()
        # This corresponds to one Mutation per Case
        # m_type = random.choice(MutationMode, size=(events.shape[0], 5), p=self.mutation_rate.probs)
        mutation = random.random(size=(events.shape + (len(MutationMode), ))) < self.mutation_rate.probs[None, None]
        # m_type[m_type[:, 4] == MutationMode.NONE] = MutationMode.NONE

        insert_mask = self.create_insert_mask(events, mutation[..., MutationMode.INSERT])
        events, features = self.insert(events, features, insert_mask)

        delete_mask = self.create_delete_mask(events, mutation[..., MutationMode.DELETE])
        events, features = self.delete(events, features, delete_mask)

        change_mask = self.create_change_mask(events, mutation[..., MutationMode.CHANGE])
        events, features = self.substitute(events, features, change_mask)

        transp_mask = self.create_transp_mask(events, mutation[..., MutationMode.TRANSP])
        events, features = self.transpose(events, features, transp_mask)

        tmp = []
        tmp.extend(0 * np.ones(delete_mask.sum()))
        tmp.extend(1 * np.ones(change_mask.sum()))
        tmp.extend(2 * np.ones(insert_mask.sum()))
        # tmp.extend(3 * np.ones(transp_mask.sum()))
        # tmp.extend(4 * np.ones(np.sum([np.sum(~m) for m in [delete_mask, change_mask, insert_mask, transp_mask]])))
        mutations = np.array(tmp)
        return EvaluatedCases(events, features).set_mutations(mutations).evaluate_viability(self.fitness_function, fa_seed)

    def delete(self, events, features, delete_mask):
        events, features = events.copy(), features.copy()
        events[delete_mask] = 0
        features[delete_mask] = 0
        return events, features

    def substitute(self, events: np.ndarray, features: np.ndarray, change_mask: np.ndarray):
        events, features = events.copy(), features.copy()
        events[change_mask] = random.integers(1, self.vocab_len, events.shape)[change_mask]
        features[change_mask] = random.standard_normal(features.shape)[change_mask]
        return events, features

    def insert(self, events: np.ndarray, features: np.ndarray, insert_mask: np.ndarray):
        events, features = events.copy(), features.copy()
        events[insert_mask] = random.integers(1, self.vocab_len, events.shape)[insert_mask]
        features[insert_mask] = random.standard_normal(features.shape)[insert_mask]
        return events, features

    def transpose(self, events: np.ndarray, features: np.ndarray, swap_mask: np.ndarray, **kwargs):
        events, features = events.copy(), features.copy()
        is_reverse = kwargs.get('is_reverse', random.uniform() > 0.5)
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

    def create_delete_mask(self, events: np.ndarray, m_type: MutationMode) -> np.ndarray:
        delete_mask = m_type & (events != 0)
        return delete_mask

    def create_change_mask(self, events: np.ndarray, m_type: MutationMode) -> np.ndarray:
        change_mask = m_type & (events != 0)
        return change_mask

    def create_insert_mask(self, events: np.ndarray, m_type: MutationMode) -> np.ndarray:
        insert_mask = m_type & (events == 0)
        return insert_mask

    def create_transp_mask(self, events: np.ndarray, m_type: MutationMode) -> np.ndarray:
        transp_mask = m_type & (np.ones_like(events) == 1)
        transp_mask[:, [0, 1, -1, -2]] = False
        return transp_mask

    def set_data_distribution(self, data_distribution: DataDistribution) -> RandomMutator:
        self.data_distribution = data_distribution
        return self


class SamplingBasedMutator(RandomMutator):
    def set_data_distribution(self, data_distribution: DataDistribution) -> SamplingBasedMutator:
        return super().set_data_distribution(data_distribution)

    def insert(self, events: np.ndarray, features: np.ndarray, insert_mask: np.ndarray):
        events, features = events.copy(), features.copy()
        dist: DataDistribution = self.data_distribution
        changed_sequences = np.any(insert_mask, axis=1)  # Only changed sequences need new features
        if changed_sequences.any():
            events[insert_mask] = random.integers(1, self.vocab_len, events.shape)[insert_mask]
            sampled_features = dist.sample_features(events[changed_sequences])
            features[changed_sequences] = sampled_features
        return events, features

    def substitute(self, events: np.ndarray, features: np.ndarray, change_mask: np.ndarray):
        events, features = events.copy(), features.copy()
        dist: DataDistribution = self.data_distribution
        changed_sequences = change_mask.any(axis=1)  # Only changed sequences need new features
        if changed_sequences.any():
            events[change_mask] = random.integers(1, self.vocab_len, events.shape)[change_mask]
            sampled_features = dist.sample_features(events[changed_sequences])
            features[changed_sequences] = sampled_features
        return events, features


class Recombiner(EvolutionaryOperatorInterface, ABC):
    recombination_rate: Number = None

    @abstractmethod
    def recombination(self, cf_offspring: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        pass

    def set_recombination_rate(self, recombination_rate: float) -> Recombiner:
        self.recombination_rate = recombination_rate
        return self

    def get_config(self):
        return BetterDict(super().get_config()).merge({'recombiner': {'type': type(self).__name__, 'recombination_rate': self.recombination_rate}})


class FittestSurvivorRecombiner(Recombiner):
    def recombination(self, cf_mutated: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_offspring = cf_mutated + cf_population
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        selector = (fitness.viabs > np.median(fitness.viabs)).flatten()
        # selector = ranking[-self.num_survivors:].flatten()
        selected_fitness = fitness[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]

        selected = EvaluatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        sorted_selected = (cf_population + selected).sort().get_topk(self.num_survivors)
        return sorted_selected


class BestBreedRecombiner(Recombiner):
    def recombination(self, cf_mutated: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_ev_offspring, cf_ft_offspring, _, offspring_fitness = cf_mutated.all
        # mutations = cf_offspring.mutations
        selector = (offspring_fitness.viabs > np.median(offspring_fitness.viabs)).flatten()
        selected_fitness = offspring_fitness[selector]
        selected_events = cf_ev_offspring[selector]
        selected_features = cf_ft_offspring[selector]
        # selected_mutations = mutations[selector]

        selected_offspring = EvaluatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        
        sorted_selected = cf_population.sort().get_topk(self.num_survivors-len(selected_offspring))

        return selected_offspring + sorted_selected


class RankedRecombiner(Recombiner):
    def recombination(self, cf_mutated: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_offspring = cf_mutated + cf_population
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        M = Viabilities.Measures
        importance = [M.DATA_LLH, M.OUTPUT_LLH, M.SPARCITY, M.SIMILARITY, M.MODEL_LLH]
        parts = fitness._parts[importance][..., 0].T
        tmp: pd.DataFrame = pd.DataFrame(parts).sort_values([0, 1, 2, 3], ascending=False)
        selector = tmp.index.values

        # selector = ranking[-self.num_survivors:].flatten()
        selected_fitness = fitness[selector]
        selected_events = cf_ev[selector]
        selected_features = cf_ft[selector]

        selected = EvaluatedCases(selected_events, selected_features, selected_fitness)  #.set_mutations(selected_mutations)
        sorted_selected = selected[:self.num_survivors]
        return sorted_selected


class RankedParetoRecombiner(HierarchicalMixin, Recombiner):
    def recombination(self, cf_mutated: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_offspring = cf_mutated + cf_population
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        selected = self.order(cf_offspring)

        sorted_selected = selected[:self.num_survivors]
        return sorted_selected


class ParetoRecombiner(ParetoMixin, Recombiner):
    def recombination(self, cf_mutated: EvaluatedCases, cf_population: EvaluatedCases, **kwargs) -> EvaluatedCases:
        cf_offspring = cf_mutated + cf_population
        cf_ev, cf_ft, _, fitness = cf_offspring.all
        selected = self.order(cf_offspring)

        sorted_selected = selected[:self.num_survivors]
        return sorted_selected


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
        crossover_rate = kwargs.get('crossover_rate', 0.2)
        edit_rate = kwargs.get('edit_rate', 0.1)
        mutation_rate = kwargs.get('mutation_rate', MutationRate())
        recombination_rate = kwargs.get('recombination_rate', 0.5)
        initiators = initiators or [
            FactualInitiator(),
            RandomInitiator(),
            CaseBasedInitiator().set_vault(evaluator.data_distribution),
            SamplingBasedInitiator().set_data_distribution(evaluator.measures.dllh.data_distribution),
        ]
        selectors = selectors or [
            RouletteWheelSelector(),
            TournamentSelector(),
            ElitismSelector(),
        ]
        crossers = crossers or [
            # OnePointCrosser(),
            TwoPointCrosser(),
            UniformCrosser().set_crossover_rate(crossover_rate),
        ]
        mutators = mutators or [
            # DefaultMutator().set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
            # RestrictedDeleteInsertMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
            SamplingBasedMutator().set_data_distribution(evaluator.measures.dllh.data_distribution).set_mutation_rate(mutation_rate).set_edit_rate(edit_rate),
        ]
        recombiners = recombiners or [
            FittestSurvivorRecombiner(),
            BestBreedRecombiner(),
        ]

        combos = itertools.product(initiators, selectors, crossers, mutators, recombiners)
        result = [EvoConfigurator(*cnf) for cnf in combos]
        return result