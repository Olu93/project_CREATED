from thesis_viability.viability.viability_function import ViabilityMeasure
from thesis_generators.models.evolutionary_strategies.skeleton import EvolutionaryStrategy
import numpy as np


class UnrestricatedEvStrategy(EvolutionaryStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _init_population(self, fc_seed, **kwargs):
        fc_ev, fc_ft = fc_seed
        random_events = np.random.randint(0, self.vocab_len, fc_ev.shape)
        random_features = np.random.randn(0, self.vocab_len, fc_ft.shape)
        return random_events, random_features

    def _recombine_parents(self, events, features, total, *args, **kwargs):
        # Parent can mate with itself, as that would preserve some parents
        mother_ids, father_ids = np.random.randint(0, len(events), (2, total))
        mother_events, father_events = events[mother_ids], events[father_ids]
        mother_features, father_features = features[mother_ids], features[father_ids]
        gene_flips = np.random.random((total, mother_events.shape[1])) > 0.5
        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]
        return child_events, child_features

    def _mutate_offspring(self, cf_offspring, factual_seed, *args, **kwargs):
        cf_ev, cf_ft = cf_offspring

    def _mutate_events(self, events, features, mutation_rate, *args, **kwargs):
        # mutation_selection = np.random.random([events.shape[0]]) > mutation_rate
        m_type = np.random.randint(0, 4, events.shape[0])
        # TODO: Use EnumInt

        # DELETE
        events[m_type == 0] = 0
        features[m_type == 0] = 0
        # CHANGE
        events[m_type == 1] = np.random.randint(1, self.vocab_len, events.shape)[m_type == 1]
        features[m_type == 1] = np.random.randn(features.shape)[m_type == 1]
        # INSERT
        events[(m_type == 2) & (events == 0)] = np.random.randint(1, self.vocab_len, events.shape)[(m_type == 2) & (events == 0)]
        features[m_type == 2 & (events == 0)] = np.random.randn(features.shape)[m_type == 2 & (events == 0)]
        # SWAP
        swap_mask = (m_type == 3) & (np.random.random([events.shape[0]]) > 0.1)
        
        source_container = np.roll(events, -1, axis=1)
        tmp_container = np.ones_like(events) * np.nan
        tmp_container[swap_mask] = events[swap_mask]
        tmp_container = np.roll(source_container, 1, axis=1)
        backswap_mask = ~np.isnan(tmp_container)
        
        events[swap_mask] = source_container[swap_mask]
        events[backswap_mask] = tmp_container[backswap_mask]
        
        tmp_container = np.ones_like(features) * np.nan
        tmp_container[swap_mask] = events[swap_mask]
        tmp_container = np.roll(source_container, 1, axis=1)
        
        features[swap_mask] = source_container[swap_mask]
        features[backswap_mask] = tmp_container[backswap_mask]
        
        return events, features
    

    def _generate_population(self, cf_parents, factual_seed, **kwargs):
        cf_ev, cf_ft = cf_parents
        offspring = self._recombine_parents(cf_ev, cf_ft, self.num_population)
        return offspring

    def determine_fitness(self, cf_offspring, fc_seed, **kwargs):
        cf_ev, cf_ft = cf_offspring
        fc_ev, fc_ft = fc_seed
        fitness = self.fitness_function(fc_ev, fc_ft, cf_ev, cf_ft)
        return fitness
