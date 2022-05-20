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
        gene_flips = np.random.random((total, mother_events.shape[1]))
        child_events = mother_events.copy()
        child_events[gene_flips] = father_events[gene_flips]
        child_features = mother_features.copy()
        child_features[gene_flips] = father_features[gene_flips]
        return child_events, child_features


    def _mutate_offspring(self, cf_offspring, factual_seed, *args, **kwargs):
        return

    def _generate_population(self, cf_parents, factual_seed, **kwargs):
        cf_ev, cf_ft = cf_parents
        offspring = self._recombine_parents(cf_ev, cf_ft, self.num_population)
        return offspring

    def determine_fitness(self, cf_offspring, fc_seed, **kwargs):
        cf_ev, cf_ft = cf_offspring
        fc_ev, fc_ft = fc_seed
        fitness = self.fitness_function(fc_ev, fc_ft, cf_ev, cf_ft)
        return fitness
    

