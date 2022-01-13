from collections import defaultdict
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from tensorflow.python.keras.metrics import CategoricalAccuracy
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from tqdm import tqdm
import textdistance
from tensorflow.keras import Model
from ..models.model_commons import ModelInterface
from thesis_readers.helper.modes import TaskModeType, TaskModes
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from ..helper.constants import NUMBER_OF_INSTANCES, SEQUENCE_LENGTH
from ..models.lstm import SimpleLSTMModelOneWayExtensive
from ..models.transformer import TransformerModelOneWayExtensive
from thesis_readers.readers.BPIC12LogReader import BPIC12LogReader

STEP1 = "Step 1: Iterate through data"
STEP2 = "Step 2: Compute Metrics"
STEP3 = "Step 3: Save results"
FULL = 'FULL'
symbol_mapping = {index: char for index, char in enumerate(set([chr(i) for i in range(1, 3000) if len(chr(i)) == 1]))}


class Evaluator(object):
    def __init__(self, reader: AbstractProcessLogReader) -> None:
        super().__init__()
        self.reader = reader
        self.idx2vocab = self.reader.idx2vocab
        self.task_mode_type = None

    def set_task_mode(self, mode: TaskModes):
        self.task_mode_type = mode
        return self

    def set_model(self, model: ModelInterface):
        self.model = model
        self.task_mode_type = self.model.task_mode_type
        return self

    def evaluate(self, test_dataset: DatasetV2, metric_mode='weighted'):
        test_dataset_full = self.reader.gather_full_dataset(test_dataset)
        if self.task_mode_type == TaskModeType.FIX2ONE:
            return self.results_simple(test_dataset_full, metric_mode)
        if self.task_mode_type == TaskModeType.FIX2FIX:
            return self.results_extensive(test_dataset_full, metric_mode)

    def results_extensive(self, test_dataset, mode='weighted'):
        print("Start results by instance evaluation")
        print(STEP1)
        X_test, y_test = test_dataset
        y_test = y_test.astype(int)
        print(STEP2)
        eval_results = []
        y_pred = self.model.predict(X_test[0] if len(X_test) == 1 else X_test).argmax(axis=-1).astype(np.int32)
        iterator = enumerate(zip(X_test[0], y_test, y_pred))
        for idx, (row_x_test, row_y_test, row_y_pred) in tqdm(iterator, total=len(y_test)):
            row_x_test = row_x_test.astype(np.int32)
            first_word_test = np.max([np.argmin(row_y_test == 0), 1])
            first_word_pred = np.max([np.argmin(row_y_pred == 0), 1])
            first_word_x = np.argmin(row_x_test == 0)
            longer_sequence_start = min([first_word_test, first_word_pred])
            instance_result = {
                "trace": idx,
                f"full_{SEQUENCE_LENGTH}": len(row_y_test) - first_word_test,
                f"input_x_{SEQUENCE_LENGTH}": len(row_x_test) - first_word_x,
                f"true_y_{SEQUENCE_LENGTH}": len(row_y_test) - first_word_test,
                f"pred_y_{SEQUENCE_LENGTH}": len(row_y_pred) - first_word_pred,
            }
            instance_result.update(self.compute_traditional_metrics(mode, row_y_test[longer_sequence_start:], row_y_pred[longer_sequence_start:]))
            instance_result.update(self.compute_sequence_metrics(row_y_test[first_word_test:], row_y_pred[-first_word_pred:]))
            instance_result.update(self.compute_decoding(row_y_pred[first_word_pred:], row_y_test[first_word_test:], row_x_test[first_word_x:]))
            eval_results.append(instance_result)

        results = pd.DataFrame(eval_results)
        print(STEP3)
        print(results)
        return results

    def results_simple(self, test_dataset, mode='weighted'):
        print("Start results by instance evaluation")
        print(STEP1)
        X_test, y_test = test_dataset
        y_test = y_test.astype(int).reshape(-1)
        X_test = X_test[0] if len(X_test) == 1 else X_test
        print(STEP2)
        y_pred = self.model.predict(X_test).argmax(axis=-1).astype(np.int32)
        x_test_rows = [tuple(x) for x in (X_test[0] if len(X_test) == 2 else X_test)]
        df = pd.DataFrame()
        df["trace"] = range(len(x_test_rows))
        df[f"input_x_{SEQUENCE_LENGTH}"] = [np.not_equal(x, 0).sum() for x in x_test_rows]
        df[f"pred_y"] = y_pred
        df[f"true_y"] = y_test
        df[f"is_correct"] = y_pred == y_test

        print(STEP3)
        print(df)
        return df

    def compute_traditional_metrics(self, mode, row_y_test_zeros, row_y_pred_zeros):
        return {
            "acc": accuracy_score(row_y_pred_zeros, row_y_test_zeros),
            "recall": recall_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
            "precision": precision_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
            "f1": f1_score(row_y_pred_zeros, row_y_test_zeros, average=mode, zero_division=0),
        }

    def compute_sequence_metrics(self, true_seq, pred_seq):
        true_seq_symbols = "".join([symbol_mapping[idx] for idx in true_seq])
        pred_seq_symbols = "".join([symbol_mapping[idx] for idx in pred_seq])
        dict_instance_distances = {
            "levenshtein": textdistance.levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "damerau_levenshtein": textdistance.damerau_levenshtein.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "local_alignment": textdistance.smith_waterman.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "global_alignment": textdistance.needleman_wunsch.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "emph_start": textdistance.jaro_winkler.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "longest_subsequence": textdistance.lcsseq.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "longest_substring": textdistance.lcsstr.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "overlap": textdistance.overlap.normalized_similarity(true_seq_symbols, pred_seq_symbols),
            "entropy": textdistance.entropy_ncd.normalized_similarity(true_seq_symbols, pred_seq_symbols),
        }
        return dict_instance_distances

    def compute_decoding(self, row_y_pred, row_y_test, row_x_test):
        x_convert = [f"{i:03d}" for i in row_x_test]
        return {
            "input": " | ".join(["-".join(x_convert[:lim + 1]) for lim in range(len(x_convert))]),
            "true_encoded": " -> ".join([f"{i:03d}" for i in row_y_test]),
            "pred_encoded": " -> ".join([f"{i:03d}" for i in row_y_pred]),
            "true_encoded_with_padding": " -> ".join([f"{i:03d}" for i in row_y_test]),
            "pred_encoded_with_padding": " -> ".join([f"{i:03d}" for i in row_y_pred]),
            "true_decoded": " -> ".join([self.idx2vocab[i] for i in row_y_test]),
            "pred_decoded": " -> ".join([self.idx2vocab[i] for i in row_y_pred]),
        }


if __name__ == "__main__":
    data = BPIC12LogReader(debug=False)
    data = data.init_log(True)
    data = data.init_data()
    train_dataset = data.get_dataset().take(1000)
    val_dataset = data.get_val_dataset().take(100)
    test_dataset = data.get_test_dataset()

    model = SimpleLSTMModelOneWayExtensive(data.vocab_len, data.max_len)
    # model = TransformerModel(data.vocab_len, data.max_len)
    model.build((None, data.max_len))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=[CategoricalAccuracy()])
    model.summary()

    model.fit(train_dataset, batch_size=100, epochs=1, validation_data=val_dataset)

    results_by_instance_seq2seq(data.idx2vocab, data.start_id, data.end_id, test_dataset, model, 'junk/test1_.csv')
    # results_by_len(data.idx2vocab, test_dataset, model, 'junk/test2_.csv')
    # show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test3_.csv', mode=None)
    # show_predicted_seq(data.idx2vocab, test_dataset, model, save_path='junk/test4_.csv', mode=FULL)
