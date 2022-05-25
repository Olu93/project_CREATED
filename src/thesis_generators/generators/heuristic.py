from typing import Union
from thesis_commons.functions import shift_seq_forward, shift_seq_backward
from thesis_commons.modes import TaskModes, FeatureModes
from thesis_readers import AbstractProcessLogReader
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
import textdistance
from nltk.lm.models import LanguageModel
from thesis_readers import DomesticDeclarationsLogReader as Reader

from thesis_generators.helper.constants import SYMBOL_MAPPING
from thesis_generators.helper.wrapper import ModelWrapper

np.set_printoptions(edgeitems=26, linewidth=100000)

# PAPER: Aprroach does not work with multivariate data
class HeuristicGenerator():

    def __init__(self, reader: AbstractProcessLogReader, model_wrapper: ModelWrapper, threshold: float = 0.8):
        self.reader = reader
        self.model_wrapper = model_wrapper
        self.threshold = threshold
        self.longest_sequence = self.reader.max_len
        self.num_states = self.reader.vocab_len
        self.end_id = self.reader.end_id
        self.start_id = self.reader.start_id
        self.pad_id = self.reader.pad_id
        self.lm_hard = self.reader.trace_ngrams_hard
        self.lm_soft = self.reader.trace_ngrams_soft

    def generate_counterfactual_outcome(self, true_seq: np.ndarray, true_outcome: int, desired_outcome: int) -> np.ndarray:
        counter_factual_candidates = np.array(true_seq, dtype=int)
        pred_index = desired_outcome
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
                prob_of_desired_outcome = predictions[:, pred_index]
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
                pred_index = most_likely_index
                if most_likely_index == self.reader.start_id or most_likely_index == 0:
                    break
        print("Done")

    def generate_counterfactual_next(self, true_seq: np.ndarray, true_outcome: int, desired_outcome: int) -> np.ndarray:
        pred_index = desired_outcome
        runs = {}
        for idx, seq in enumerate(np.array(true_seq, dtype=int)):
            all_candidates = []
            print("Candidate sequence")
            print(seq)
            print("Candidate sequence outcome")
            print(true_outcome)
            counterfactual_candidate = np.array(seq)
            # counterfactual_candidate[-1] = desired_outcome
            # counterfactual_candidate[:-1] = seq[1:]
            tmp_candidate = np.array(counterfactual_candidate)
            len_candidate = self.longest_sequence
            num_events = np.count_nonzero(counterfactual_candidate)
            stop_idx = len_candidate - num_events
            min_prob = self.model_wrapper.prediction_model.predict(tmp_candidate[None].astype(np.float32))[0, desired_outcome]
            all_candidates.extend(self.find_all_probable_backwards(tmp_candidate[None], len_candidate - 1, 0.2, np.array([desired_outcome])[None, None], stop_idx))
            array_all_candidates = np.array(all_candidates)
            predictions = self.model_wrapper.prediction_model.predict(array_all_candidates.astype(np.float32))
            probabilities_for_desired_outcome = predictions[:, desired_outcome]
            # # True version
            seq_ranking = probabilities_for_desired_outcome.argsort()[::-1]
            ordered_traces = array_all_candidates[seq_ranking]
            ordered_model_probs = probabilities_for_desired_outcome[seq_ranking]
            ordered_ngram_probs = self.compute_ngram_probabilities(ordered_traces, self.lm_hard, self.reader)
            ordered_ngram_probs_soft = self.compute_ngram_probabilities(ordered_traces, self.lm_soft, self.reader)
            runs[idx] = {"sequences": ordered_traces, "model_probs": ordered_model_probs, "ngram_probs_hard": ordered_ngram_probs, "ng_probs_soft": ordered_ngram_probs_soft}
            print("--- Results ---")
            for k, v in runs[idx].items():
                print(k)
                print(v)
            print("Valid sequences")
            print(runs[idx]["sequences"][runs[idx]["ngram_probs_hard"]!=0])

        print("Done")

        return runs

    def compute_ngram_probabilities(self, sequences: np.ndarray, probability_estimator: LanguageModel, reader: AbstractProcessLogReader):
        sequences_to_decode = shift_seq_backward(sequences)
        sequences_to_decode[:, -1] = self.end_id
        sequences_without_padding = [row[np.nonzero(row)] for row in sequences_to_decode]
        decoded_sequences = reader.decode_matrix_str(sequences_without_padding)
        ngram_probabilities = [[probability_estimator.score(row[idx + 1], row[idx:idx + 1]) for idx in range(len(row) - 1)] for row in decoded_sequences]
        sequence_probabilities = [np.prod(row) for row in ngram_probabilities]
        return np.array(sequence_probabilities)

    def find_all_probable_forward(self, candidates, idx, min_prob, desired_outcomes, stop_idx):
        candidates_to_check = np.array(candidates)
        target = candidates[:, -1]
        to_be_shifted = candidates
        while True:
            shifted_candidates = shift_seq_forward(to_be_shifted)
            is_empty_seq = shifted_candidates[:, :-1].sum() == 0
            if is_empty_seq:
                break
            shifted_candidates[:, -1] = target
            candidates_to_check = np.vstack([candidates_to_check, shifted_candidates])
            to_be_shifted = shifted_candidates
        candidates_to_check = np.unique(candidates_to_check, axis=0)
        predictions = self.model_wrapper.prediction_model.predict(candidates_to_check.astype(np.float32))
        fitting_probs = np.take_along_axis(predictions, desired_outcomes[..., 0], axis=1)
        to_pick = (fitting_probs > min_prob)
        result = candidates_to_check[to_pick.flatten()]
        return result

    def find_all_probable_backwards(self, candidates, idx, min_prob, desired_outcomes, stop_idx):
        if idx <= stop_idx:
            # print(f"========== {idx} ===========")
            return candidates
        if idx == stop_idx + 1:
            print("Stop right here")
        print(f"processing... {idx} - {len(candidates)}")
        options = np.roll(candidates, candidates.shape[1] - idx - 1, -1)
        options[:, :candidates.shape[1] - idx - 1] = 0
        options = np.repeat(options[:, None], self.num_states, axis=1)
        options[:, :, -1] = range(0, self.num_states)
        prediction_candidates = options.reshape((-1, options.shape[-1]))
        # TODO: Forward beam all starting with 19 by searching all starting points that end in 8
        # Shift whole true sequence to left and move one forward each

        # NOTES: There are two ways: 1. Only take possible outcomes 2. Take outcomes that increase the likelihoods
        predictions = self.model_wrapper.prediction_model.predict(prediction_candidates.astype(np.float32))
        options_probs = predictions.reshape((*options.shape[:2], -1))
        # Three sets of possible routes:
        # 1. Those that lead to desired outcome
        max_paths_aligned_with_desired_outcome = (options_probs.argmax(-1) == desired_outcomes.max(-1))
        possible_paths = options[max_paths_aligned_with_desired_outcome]

        # 2. Those that increase the odds of ending in outcome
        fitting_probs = np.take_along_axis(options_probs, desired_outcomes, axis=2)
        # new_min_prob = np.mean(fitting_probs,1)[..., None]
        better_paths = (fitting_probs > min_prob)
        mask_seq_forward = better_paths
        non_zero_positions = np.vstack(np.nonzero(mask_seq_forward)).T
        continuations = ~np.isin(non_zero_positions[:, 1], [self.start_id, self.end_id, self.pad_id])
        idx_continuations = non_zero_positions[continuations]
        # 3. Those that do neither

        if np.any(continuations):
            # TODO: Only if they end with 19
            new_candidates = options[idx_continuations.T[0], idx_continuations.T[1]]
            new_candidates_probs = fitting_probs[idx_continuations.T[0], idx_continuations.T[1]]
            new_min_prob = np.mean(new_candidates_probs)
            backward_candidates = self.find_all_probable_forward(new_candidates, idx, new_min_prob, desired_outcomes, stop_idx)
            results = self.find_all_probable_backwards(backward_candidates, idx - 1, np.max(new_candidates_probs), desired_outcomes, stop_idx)
            is_done = np.any(results == self.start_id, axis=1)
            unfinished_results = results[~is_done]
            finished_results = results[is_done]

            return finished_results

        if not np.any(continuations):
            return candidates

    def _reduction_step_random(self, candidates, predictions, max_samples, desired_outcome):
        if len(candidates) > max_samples:
            random_indices = np.random.choice(len(candidates), max_samples, False)
            return candidates[random_indices]
        return candidates

    def _reduction_step_topk(self, candidates, predictions, max_samples, desired_outcome):
        if len(candidates) > max_samples:
            order = predictions.argsort()[::-1]

            return candidates[order[:max_samples]]
        return candidates

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
    reader = Reader(mode=task_mode).init_meta()
    idx = 1
    sample = next(iter(reader.get_dataset(ft_mode=FeatureModes.FULL).batch(15)))

    example_sequence, true_outcome = sample[0][idx], sample[1][idx]
    predictor = ModelWrapper(reader).load_model_by_name("result_next_token_to_class_bi_lstm")  # 1
    generator = HeuristicGenerator(reader, predictor)
    # print(example[0][0])
    # example_sequence = np.array('0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 19 1 14'.split(), dtype=np.int32)[None]
    # true_outcome = np.array([[8]])
    # example_sequence = np.array('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19 1 11 16'.split(), dtype=np.int32)[None]
    # true_outcome = np.array([[1]])
    # example_sequence = np.array('0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19 1 11 2 3'.split(), dtype=np.int32)[None]
    # true_outcome = np.array([[3]])
    # example_sequence = np.array('0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 19  1  2  3  4'.split(), dtype=np.int32)[None]
    # true_outcome = np.array([[3]])
    generator.generate_counterfactual_next(example_sequence, true_outcome, 8)

# NOTES
# Most important is to find the lead up
# Afterwards find the sequence that maximizes the lead up
# Knowing the lead up it is one can predict the precedence
# --> Knowing the build up makes one capable to predict the precedence