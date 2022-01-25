from typing import Union
from thesis_readers.helper.modes import TaskModes, FeatureModes
from thesis_readers import AbstractProcessLogReader
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
import textdistance
from thesis_readers import DomesticDeclarationsLogReader as Reader

from thesis_generators.helper.constants import SYMBOL_MAPPING
from thesis_generators.predictors.wrapper import ModelWrapper


class HeuristicGenerator():
    def __init__(self, reader: AbstractProcessLogReader, model_wrapper: ModelWrapper, threshold: float = 0.8):
        self.reader = reader
        self.model_wrapper = model_wrapper
        self.threshold = threshold
        self.longest_sequence = self.reader.max_len
        self.num_states = self.reader.vocab_len
        self.end_id = self.reader.end_id

    # def generate_counterfactual(self, true_seq: np.ndarray, desired_outcome: int) -> np.ndarray:
    #     cutoff_points = tf.argmax((true_seq == self.end_id), -1) - 1
    #     counter_factual_candidate = np.array(true_seq, dtype=int)
    #     counter_factual_candidate[:, cutoff_points] = desired_outcome
    #     num_edits = 5
    #     for row_num, row in enumerate(counter_factual_candidate):
    #         for edit_num in range(num_edits):
    #             cut_off = cutoff_points[row_num]
    #             true_seq_cut = row[:]
    #             top_options = np.zeros((cut_off, self.longest_sequence))
    #             top_probabilities = (-np.ones(cut_off) * np.inf)
    #             for step in reversed(range(1, cut_off)):  # stack option sets
    #                 options = np.repeat(row[None], self.num_states - 4, axis=0)
    #                 options[:, step] = range(2, self.num_states - 2)
    #                 # print(f"------------- Run {row_num} ------------")
    #                 # print("Candidates")
    #                 # print(options)
    #                 predictions = self.model_wrapper.predict_sequence(options)
    #                 # print("Predictions")
    #                 # print(predictions.argmax(-1))

    #                 # print(np.product(predictions[0, :, options[0]], axis=-1))
    #                 indices = options.reshape(-1, 1)
    #                 flattened_preds = predictions.reshape(-1, predictions.shape[-1])
    #                 multipliers = np.take_along_axis(flattened_preds, indices, axis=1).reshape(predictions.shape[0], -1)
    #                 # seq_probs = np.prod(multipliers[:, :cut_off+1], -1)
    #                 seq_probs = np.log(multipliers[:, :]).sum(axis=-1)
    #                 seq_probs_rank = np.argsort(seq_probs)[::-1]
    #                 options_ranked = options[seq_probs_rank, :]
    #                 seq_probs_ranked = seq_probs[seq_probs_rank]
    #                 all_metrics = [self.compute_sequence_metrics(true_seq_cut, cf_seq_cut) for cf_seq_cut in options_ranked]
    #                 # selected_metrics = [idx for idx, metrics in enumerate(all_metrics) if metrics['damerau_levenshtein'] >= self.threshold]
    #                 # viable_candidates = options_ranked[selected_metrics]
    #                 viable_candidates = options_ranked
    #                 # print("Top")
    #                 # print(viable_candidates[:5])
    #                 top_options[cut_off - step - 1] = viable_candidates[0]
    #                 top_probabilities[cut_off - step - 1] = seq_probs_ranked[0]
    #                 # print("Round done")
    #             print(f"TOP RESULTS")
    #             print(top_options)
    #             print(top_probabilities)
    #             print(top_probabilities.argmax())
    #             print(top_options[top_probabilities.argmax()])
    #             row = top_options[top_probabilities.argmax()].astype(int)
    #     print("Done")

    # def compute_backwards_probs(self, options, position):
    #     counter_factual_candidate

    def generate_counterfactual(self, true_seq: np.ndarray, true_outcome: int, desired_outcome: int) -> np.ndarray:
        counter_factual_candidates = np.array(true_seq, dtype=int)
        for seq in counter_factual_candidates:
            candidate = np.array(seq)
            # most_likely_index = -42
            for i in reversed(range(candidate.shape[-1])):
                print(f"========== {i} ===========")
                print(f"Candidate: {candidate}")
                options = np.repeat(candidate[None], self.num_states, axis=0)
                options[:, i] = range(0, self.num_states)
                # zeros_mask = counter_factual_candidate != 0
                predictions = self.model_wrapper.prediction_model.predict(options.astype(np.float32))
                prob_of_desired_outcome = predictions[:, desired_outcome]
                # indices = options.reshape(-1, 1)
                # flattened_preds = predictions.reshape(-1, predictions.shape[-1])
                # multipliers = np.take_along_axis(flattened_preds, indices, axis=1).reshape(predictions.shape[0], -1)
                # results = results[zeros_mask]
                most_likely_index = prob_of_desired_outcome.argmax()
                most_likely_sequence = options[most_likely_index]
                most_likely_prob = prob_of_desired_outcome[most_likely_index]
                print(most_likely_index)
                print(most_likely_prob)
                print(most_likely_sequence)
                candidate = most_likely_sequence
                if most_likely_index == self.reader.start_id or most_likely_index == 0:
                    break
        print("Done")
                
            
        


    def compute_sequence_metrics(self, true_seq: np.ndarray, counterfactual_seq: np.ndarray):
        true_seq_symbols = "".join([SYMBOL_MAPPING[idx] for idx in true_seq])
        cfact_seq_symbols = "".join([SYMBOL_MAPPING[idx] for idx in counterfactual_seq])
        dict_instance_distances = {
            "levenshtein": textdistance.levenshtein.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "damerau_levenshtein": textdistance.damerau_levenshtein.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "local_alignment": textdistance.smith_waterman.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "global_alignment": textdistance.needleman_wunsch.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "emph_start": textdistance.jaro_winkler.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "longest_subsequence": textdistance.lcsseq.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "longest_substring": textdistance.lcsstr.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "overlap": textdistance.overlap.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
            "entropy": textdistance.entropy_ncd.normalized_similarity(true_seq_symbols, cfact_seq_symbols),
        }
        return dict_instance_distances


if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME
    reader = Reader(mode=task_mode).init_data()
    idx = 0
    sample = next(iter(reader.get_dataset(ft_mode=FeatureModes.EVENT_ONLY).batch(15)))
    example_sequence, true_outcome = sample[0][idx], sample[1][idx] 
    predictor = ModelWrapper(reader).load_model_by_name("result_outcome_token_to_class_bi_lstm")  # 1
    generator = HeuristicGenerator(reader, predictor)
    # print(example[0][0])

    print(generator.generate_counterfactual(example_sequence, true_outcome, 8))  # 6, 15, 18 | 8
