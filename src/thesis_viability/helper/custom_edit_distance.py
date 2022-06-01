import io
import pickle
from typing import Any, Callable
from unicodedata import is_normalized

import numpy as np
import tensorflow as tf
from scipy.spatial import distance

import thesis_viability.helper.base_distances as distances
from thesis_commons.functions import stack_data
from thesis_commons.modes import DatasetModes, GeneratorModes, TaskModes
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_readers import MockReader as Reader

DEBUG_SLOW = False

class DamerauLevenshstein():

    def __init__(self, vocab_len: int, max_len: int, distance_func: distances.BaseDistance):
        self.dist = distance_func
        self.vocab_len = vocab_len
        self.max_len = max_len

    def __call__(self, s1, s2):
        s1_ev, s1_ft = s1
        s2_ev, s2_ft = s2
        s1_batch_size, s1_seq_len, s1_ft_len = s1_ft.shape
        s2_batch_size, s2_seq_len, s2_ft_len = s2_ft.shape
        
        s1_ev, s1_ft = np.repeat(s1_ev, s2_batch_size, axis=0), np.repeat(s1_ft, s2_batch_size, axis=0)
        s2_ev, s2_ft = np.repeat(s2_ev[None], s1_batch_size, axis=0).reshape(-1, s2_seq_len), np.repeat(s2_ft[None], s1_batch_size, axis=0).reshape(-1, s2_seq_len, s2_ft_len)
        
        lenstr1 = s1_ev.shape[-1]
        lenstr2 = s2_ev.shape[-1]
        num_instances = len(s1_ev)
        s1_default_distances = self.dist(s1_ft, np.zeros_like(s1_ft))  
        s2_default_distances = self.dist(s2_ft, np.zeros_like(s2_ft))  # TODO: Check max should be changed. Not zeros_lile but ones_like * BIG_CONST (-42 maybe)
        d = np.zeros((num_instances, lenstr1, lenstr2))        

        d[:, :, 0] = (np.arange(0, lenstr1) * s1_default_distances.max()).T
        d[:, 0, :] = (np.arange(0, lenstr2) * s2_default_distances.max()).T

        # TODO: Check why features have last three columns always being zero -- Needs debug mode to see it
        # TODO: Make sure this works for both sides being of differing sizes
        is_padding_symbol = ~((s1_ev != 0) | (s2_ev != 0))
        mask_s1_ev = np.ma.masked_where(is_padding_symbol, s1_ev)
        mask_s2_ev = np.ma.masked_where(is_padding_symbol, s2_ev)
        mask_s1_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(mask_s1_ev)[..., None], s1_ft_len, -1), s1_ft)
        mask_s2_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(mask_s2_ev)[..., None], s1_ft_len, -1), s2_ft)
        
        for i in range(1, lenstr1):
            for j in range(1, lenstr2):
                is_same_event = (mask_s1_ev[:, i - 1] == mask_s2_ev[:, j - 1])
                cost = is_same_event * self.dist(mask_s1_ft[:, i - 1], mask_s2_ft[:, j - 1])
                cost += ((~is_same_event) * (s1_default_distances[:, i - 1] + s2_default_distances[:, j - 1]))
                deletion = d[:, i - 1, j] + s1_default_distances[:, i - 1]
                insertion = d[:, i, j - 1] + s2_default_distances[:, j - 1]
                substitution = d[:, i - 1, j - 1] + cost
                transposition = np.ones_like(d[:, i, j]) * np.inf
                if i > 1 and j > 1:
                    one_way = mask_s1_ev[:, i - 1] == mask_s2_ev[:, j - 2]
                    bck_way = mask_s1_ev[:, i - 2] == mask_s2_ev[:, j - 1]
                    is_transposed = one_way & bck_way
                    prev_d = np.copy(d[:, i - 2, j - 2])
                    prev_d[~is_transposed] = np.inf
                    transposition = prev_d + cost

                cases = np.array([
                    deletion,
                    insertion,
                    np.ma.getdata(substitution) + np.ma.getmaskarray(substitution) * self.dist.MAX_VAL, # TODO: Use theoretical maximum of distance
                    transposition,
                ])
                min_d = np.min(cases, axis=0)
                d[:, i, j] = min_d

        if DEBUG_SLOW:
            print("------")
            print(d)
        
        all_distances = d[:, lenstr1-1, lenstr2-1]
        result = all_distances.reshape((s1_batch_size, s2_batch_size))
        self.normalizing_constants = (d[:, -1, 0] + d[:, 0, -1]).reshape((s1_batch_size, s2_batch_size))
        self.result = result
        return result
        
        


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


    loss = DamerauLevenshstein(reader.vocab_len, reader.max_len, distances.EuclidianDistance())
    # loss = DamerauLevenshstein(reader.vocab_len, reader.max_len, distances.CosineDistance())
    # loss = DamerauLevenshstein(reader.vocab_len, reader.max_len, distances.SparcityDistance())
    a_stacked = stack_data(a)
    b_stacked = stack_data(b)
    bulk_distances = loss(a_stacked, b_stacked)
    all_results = bulk_distances
    print(f"All results\n{all_results}")
    if all_results.sum() == 0:
        print("Hmm...")


