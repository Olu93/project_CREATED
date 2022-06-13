from typing import Type, TYPE_CHECKING
# if TYPE_CHECKING:
#     from thesis_generators.models.evolutionary_strategies.base_evolutionary_strategy import EvolutionaryStrategy

from thesis_commons.functions import extract_padding_mask
from thesis_commons.modes import MutationMode
from thesis_commons.representations import Cases, GaussianParams, MutatedCases, MutationRate
from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_commons.distributions import DataDistribution
import numpy as np
from thesis_commons import random
from numpy.typing import NDArray




class EvolutionaryOperatorInterface:
    name: str = "NA"
    vocab_len: int = None
    num_population: int = None
    fitness_function: ViabilityMeasure = None
    num_survivors: int = None
    mutation_rate: MutationRate = None
    recombination_rate: float = None
    edit_rate: float = None


class InitiationMixin(EvolutionaryOperatorInterface):
    pass


class SelectionMixin(EvolutionaryOperatorInterface):
    pass


class CrossoverMixin(EvolutionaryOperatorInterface):
    pass


class MutationMixin(EvolutionaryOperatorInterface):
    pass


class RecombinationMixin(EvolutionaryOperatorInterface):
    pass
   

class DefaultInitialisationMixin(InitiationMixin):
    def init_population(self, fa_seed: MutatedCases, **kwargs):
        fc_ev, fc_ft = fa_seed.cases
        random_events = random.integers(0, self.vocab_len, (self.num_population, ) + fc_ev.shape[1:]).astype(float)
        random_features = random.standard_normal((self.num_population, ) + fc_ft.shape[1:])
        return MutatedCases(random_events, random_features).evaluate_fitness(self.fitness_function, fa_seed)
    
class CasebasedInitialisationMixin(InitiationMixin):
    def init_population(self, fa_seed: MutatedCases, **kwargs):
        vault:Cases = kwargs.get('vault')
        all_cases = vault.sample(self.num_population)
        events, features = all_cases.cases
        return MutatedCases(events, features).evaluate_fitness(self.fitness_function, fa_seed)

class GaussianSampleInitializationMixin(InitiationMixin):
    def init_population(self, fa_seed: MutatedCases,**kwargs):
        dist:DataDistribution = kwargs.get('data_distribution')
        sampled_cases = dist.sample(self.num_population)
        events, features = sampled_cases.cases
        return MutatedCases(events, features).evaluate_fitness(self.fitness_function, fa_seed)


class RouletteWheelSelectionMixin(SelectionMixin):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        normalized_viabs = viabs / viabs.sum()
        selection = random.choice(np.arange(len(cf_population)), size=100, p=normalized_viabs)
        cf_selection = cf_population[selection]
        return cf_selection


class TournamentSelectionMixin(SelectionMixin):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        num_contenders = len(cf_population)
        # Seems Superior
        left_corner = random.choice(np.arange(0, num_contenders), size=self.num_population)
        right_corner = random.choice(np.arange(0, num_contenders), size=self.num_population)
        # Seems Inferior
        # left_corner = random.choice(np.arange(0, num_contenders), size=num_contenders, replace=False)
        # right_corner = random.choice(np.arange(0, num_contenders), size=num_contenders, replace=False)
        left_is_winner = viabs[left_corner] > viabs[right_corner]

        winners = np.ones_like(left_corner)
        winners[left_is_winner] = left_corner[left_is_winner]
        winners[~left_is_winner] = right_corner[~left_is_winner]

        cf_selection = cf_population[winners]
        return cf_selection


class ElitismSelectionMixin(SelectionMixin):
    # https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    def selection(self, cf_population: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        evs, fts, llhs, fitness = cf_population.all
        viabs = fitness.viabs.flatten()
        ranking = np.argsort(viabs, axis=0)
        selector = ranking[-self.num_survivors:]
        cf_selection = cf_population[selector]
        return cf_selection


class CutPointCrossoverMixin(CrossoverMixin):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = self.num_population
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = random.integers(0, len(cf_ev), (2, total))
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]

        positions = np.ones((total, mother_events.shape[1])) * np.arange(0, mother_events.shape[1])[None]
        cut_points = random.integers(0, mother_events.shape[1], size=total)[:, None]
        take_pre = random.random(size=total)[:, None] > 0.5
        gene_flips = positions > cut_points
        gene_flips[take_pre.flatten()] = ~gene_flips[take_pre.flatten()]

        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]

        return MutatedCases(child_events, child_features)


class KPointCrossoverMixin(CrossoverMixin):
    # https://www.bionity.com/en/encyclopedia/Crossover_%28genetic_algorithm%29.html
    def crossover(self, cf_parents: MutatedCases, fa_seed: MutatedCases, **kwargs) -> MutatedCases:
        cf_ev, cf_ft = cf_parents.cases
        total = self.num_population
        # Parent can mate with itself, as that would preserve some parents
        # TODO: Check this out http://www.scholarpedia.org/article/Evolution_strategies
        mother_ids, father_ids = random.integers(0, len(cf_ev), (2, total))
        mother_events, father_events = cf_ev[mother_ids], cf_ev[father_ids]
        mother_features, father_features = cf_ft[mother_ids], cf_ft[father_ids]
        mask = extract_padding_mask(mother_events)
        gene_flips = random.random((total, mother_events.shape[1])) > self.recombination_rate
        gene_flips = gene_flips & mask
        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]

        return MutatedCases(child_events, child_features)


class SingleDeleteMutationMixin(MutationMixin):
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
     
        
        return delete_mask,change_mask,insert_mask,transp_mask


class MultiDeleteMutationMixin(SingleDeleteMutationMixin):
    
    def create_mutation_masks(self, events, m_type, num_edits, positions):
        delete_mask = (m_type == MutationMode.DELETE) & (events != 0) & (positions <= num_edits)
        change_mask = (m_type == MutationMode.CHANGE) & (events != 0) & (positions <= num_edits)
        insert_mask = (m_type == MutationMode.INSERT) & (events == 0) & (positions <= num_edits)
        transp_mask = (m_type == MutationMode.TRANSP) & (positions <= num_edits)
        return delete_mask,change_mask,insert_mask,transp_mask


class DefaultRecombiner(EvolutionaryOperatorInterface):
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