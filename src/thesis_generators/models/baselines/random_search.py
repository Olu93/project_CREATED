import thesis_generators.models.model_commons as commons
import numpy as np
# https://stackoverflow.com/a/50465583/4162265
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

# https://stackoverflow.com/a/63457716/4162265
# https://stackoverflow.com/a/63991580/4162265

DEBUG_LOSS = True
DEBUG_SHOW_ALL_METRICS = True


class RandomGeneratorModel(commons.DistanceOptimizerModelMixin):
    def __init__(self, example_cases, sample_size: int = 10000, topk: int = 5, *args, **kwargs):
        print(__class__)
        super(RandomGeneratorModel, self).__init__(*args, **kwargs)
        self.example_cases = example_cases
        self.sample_size = sample_size
        self.topk = topk

    # def __call__(self, inputs):
    #     events_input, features_input = inputs
    #     batch_size, sequence_length, feature_len = features_input.shape
    #     cf_ev = np.random.randint(0, self.vocab_len, size=(self.sample_size, self.max_len))
    #     cf_ft = np.random.uniform(-5, 5, size=(self.sample_size, self.max_len, feature_len))

    #     viability_values = self.distance.compute_valuation(events_input, features_input, cf_ev, cf_ft)
    #     best_values_indices = np.argsort(viability_values, axis=1)
    #     chosen = np.where((best_values_indices >= (len(cf_ev) - self.topk)))
    #     chosen_ft_shape = (batch_size, self.topk, self.max_len, -1)
    #     chosen_ev_shape = chosen_ft_shape[:3]
    #     chosen_viab_shape = chosen_ft_shape[:2]
    #     chosen_ev_flattened, chosen_ft_flattened = cf_ev[chosen[1]], cf_ft[chosen[1]]
    #     chosen_ev, chosen_ft = chosen_ev_flattened.reshape(chosen_ev_shape), chosen_ft_flattened.reshape(chosen_ft_shape)
    #     self.chosen_viabilities = viability_values[chosen[0], chosen[1]].reshape(chosen_viab_shape)
    #     return chosen_ev, chosen_ft

    def __call__(self, inputs):
        topk = self.topk
        fa_ev, fa_ft = inputs
        cf_ev = np.random.randint(0, self.vocab_len, size=(self.sample_size, self.max_len))
        cf_ft = np.random.uniform(-5, 5, size=(self.sample_size, self.max_len, self.feature_len))
        self.picks = self.compute_topk_picks(topk, fa_ev, fa_ft, cf_ev, cf_ft)
        return self.picks['events'], self.picks['features']

    def compute_topk_picks(self, topk, fa_ev, fa_ft, cf_ev, cf_ft):
        batch_size, sequence_length, feature_len = fa_ft.shape
        viab_values, parts_values = self.compute_viabilities(fa_ev, fa_ft, cf_ev, cf_ft)
        chosen = self.pick_chosen_indices(viab_values, topk)
        shape_ev, shape_ft, shape_viab, shape_parts = self.compute_shapes(topk, batch_size, sequence_length)
        chosen_ev, chosen_ft, new_viabilities, new_partials = self.pick_topk(cf_ev, cf_ft, viab_values, parts_values, chosen, shape_ev, shape_ft, shape_viab, shape_parts)
        picks = {'events': chosen_ev, 'features': chosen_ft, 'viabilities': new_viabilities, 'partials': new_partials}
        return picks