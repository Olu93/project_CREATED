import io
from typing import Any, Callable
from unicodedata import is_normalized
import numpy as np
from thesis_commons.functions import stack_data
from thesis_viability.helper.custom_edit_distance import DamerauLevenshstein
import thesis_viability.helper.base_distances as distances
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance
import tensorflow as tf
import pickle

class SparcityMetric:
    def __init__(self, vocab_len, max_len) -> None:
        self.dist = DamerauLevenshstein(vocab_len, max_len, distances.EuclidianDistance())
        
    def compute_valuation(self, a_stacked, b_stacked):
        return self.dist(a_stacked, b_stacked)

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()

    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(1, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)

    generative_reader2 = GenerativeDataset(reader)
    train_data2 = generative_reader2.get_dataset(1, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True).shuffle(10)
    # train_data2 = generative_reader2.get_dataset(1, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)#.shuffle(10)

    a = [instances for tmp in train_data for instances in tmp]
    b = [instances for tmp in train_data2 for instances in tmp]


    sparcity_computer = SparcityMetric(reader.vocab_len, reader.max_len)

    a_stacked = stack_data(a)
    b_stacked = stack_data(b)
    bulk_distances = sparcity_computer.compute_valuation(a_stacked, b_stacked)

    print(f"All results\n{bulk_distances}")
    if bulk_distances.sum() == 0:
        print("Hmm...")


