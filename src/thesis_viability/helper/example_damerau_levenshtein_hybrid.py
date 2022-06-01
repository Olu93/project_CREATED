import io
import pickle
from typing import Any, Callable
from unicodedata import is_normalized

import numpy as np
import tensorflow as tf
from scipy.spatial import distance

from thesis_commons.modes import DatasetModes, GeneratorModes, TaskModes
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_readers import MockReader as Reader

DEBUG_SLOW = False
DEBUG_SAME_CASE = False
DEBUG_SPECIFIC_ELEMENT = 6


def levenshtein(s1, s2):
    d = {}
    s1_ev, s1_ft = s1
    s2_ev, s2_ft = s2
    lenstr1 = len(s1_ev)
    lenstr2 = len(s2_ev)
    s1_default_distances = num_changes_distance(s1_ft, np.zeros_like(s1_ft))
    s2_default_distances = num_changes_distance(s2_ft, np.zeros_like(s2_ft))
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = (i + 1) * s1_default_distances.max()
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = (j + 1) * s2_default_distances.max()
    # print("START")
    # tmp=to_matrix(d, lenstr1, lenstr2)
    # print(tmp[:-1, :-1])
    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1_ev[i] == s2_ev[j]:
                cost = num_changes_distance(s1_ft[i], s2_ft[j])
            else:
                cost = s1_default_distances[i] + s2_default_distances[j]
            d[(i, j)] = min(
                d[(i - 1, j)] + s1_default_distances[i],  # deletion
                d[(i, j - 1)] + s2_default_distances[j],  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1_ev[i] == s2_ev[j - 1] and s1_ev[i - 1] == s2_ev[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
    if DEBUG_SLOW:
        print("------")
        tmp = to_matrix(d, lenstr1, lenstr2)
        print(tmp)
    return float(d[lenstr1 - 1, lenstr2 - 1])


def to_matrix(d, lenstr1, lenstr2):
    d_tmp = np.zeros((lenstr1 + 1, lenstr2 + 1))
    for (k, l), val in d.items():
        d_tmp[k, l] = val
    return d_tmp


class DamerauLevenshstein():

    def __init__(self, vocab_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len

    def __call__(self, s1, s2, is_normalized=False):
        s1_ev, s1_ft = s1
        s2_ev, s2_ft = s2
        lenstr1 = len(s1_ev)
        lenstr2 = len(s2_ev)
        s1_default_distances = self.dist(s1_ft, np.zeros_like(s1_ft))
        s2_default_distances = self.dist(s2_ft, np.zeros_like(s2_ft))
        d = np.zeros((lenstr1 + 1, lenstr2 + 1))

        for i in range(0, lenstr1+1):
            for j in range(0, lenstr2+1):
                d[i, j] = i * s1_default_distances.max(-1) + j * s2_default_distances.max(-1)

        for i in range(1, lenstr1 + 1):
            for j in range(1, lenstr2 + 1):
                if s1_ev[i - 1] == s2_ev[j - 1]:
                    cost = self.dist(s1_ft[i - 1], s2_ft[j - 1])
                else:
                    cost = s1_default_distances[i - 1] + s2_default_distances[j - 1]
                computed_d = np.min(
                    [d[i - 1, j] + s1_default_distances[i - 1],  
                    d[i, j - 1] + s2_default_distances[j - 1],  
                    d[i - 1, j - 1] + cost,]  
                )
                d[i, j] = computed_d if not np.isnan(computed_d) else 0
                if i > 1 and j > 1:
                    if i and j and s1_ev[i - 1] == s2_ev[j - 2] and s1_ev[i - 2] == s2_ev[j - 1]:
                        d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
                # print("------")
                # print(d)
        if DEBUG_SLOW:
            print("------")
            print(d)

        return d[lenstr1, lenstr2] if not is_normalized else 1 - d[lenstr1, lenstr2] / max(lenstr1, lenstr2)


class DamerauLevenshsteinParallel():

    def __init__(self, vocab_len: int, max_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len
        self.max_len = max_len

    def __call__(self, s1, s2, is_normalized=False):
        s1_ev, s1_ft = s1
        s2_ev, s2_ft = s2
        ft_len = s1_ft.shape[-1]
        lenstr1 = self.max_len
        lenstr2 = self.max_len
        num_instances = len(s1_ev)
        s1_default_distances = self.dist(s1_ft, np.zeros_like(s1_ft))  # TODO: Check if L2 or cosine are work here, too
        s2_default_distances = self.dist(s2_ft, np.zeros_like(s2_ft))  # TODO: Check max should be changed. Not zeros_lile but ones_like * BIG_CONST (-42 maybe)
        d = np.zeros((num_instances, lenstr1, lenstr2))
        # d = np.ones((num_instances, lenstr1, lenstr2)) * 9999
        

        d[:, :, 0] = (np.arange(0, lenstr1) * s1_default_distances.max()).T
        d[:, 0, :] = (np.arange(0, lenstr2) * s2_default_distances.max()).T
        
        # for i in range(1, lenstr1):
        #     d[:,i:,i:] = np.roll(np.roll(d, 1, axis=1),1, axis=2)[:,i:,i:]
        
        # for i in range(0, self.max_len):
        #     for j in range(0, self.max_len):
        #         d[:, i, j] = i * s1_default_distances.max(-1) + j * s2_default_distances.max(-1)

        # TODO: Check why features have last three columns always being zero -- Needs debug mode to see it
        is_padding_symbol = ~((s1_ev != 0) | (s2_ev != 0))
        mask_s1_ev = np.ma.masked_where(is_padding_symbol, s1_ev)
        mask_s2_ev = np.ma.masked_where(is_padding_symbol, s2_ev)
        mask_s1_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(mask_s1_ev)[..., None], ft_len, -1), s1_ft)
        mask_s2_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(mask_s2_ev)[..., None], ft_len, -1), s2_ft)
        
        # for instance in zip(range(num_instances), is_padding_symbol.T):
        #     np.roll(np.roll(d[instance], 1, axis=0),1, axis=1)
        
        # for instance in is_padding_symbol.T:
        #     d[instance] = np.roll(np.roll(d[instance], 1),1, axis=1)
        #     d[instance, :, 0] = 0
        #     d[instance, 0, :] = 0
        
        # mask = mask_s1 & mask_s2
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
                    # if ((i-1) == 5) & ((j-2)==4):
                    #     print("Something something")

                    if np.any(is_transposed):
                        print("Something something")
                    prev_d = np.copy(d[:, i - 2, j - 2])
                    prev_d[~is_transposed] = np.inf
                    transposition = prev_d + cost

                cases = np.array([
                    deletion,
                    insertion,
                    np.ma.getdata(substitution) + np.ma.getmaskarray(substitution) * 9999, # TODO: Use theoretical maximum of distance
                    transposition,
                ])
                min_d = np.min(cases, axis=0)
                d[:, i, j] = min_d
                # if d[:, i, j].sum() == np.inf:
                #     print("Investigation time")
                # if d.sum() == np.inf:
                #     print("Investigation time")
        if DEBUG_SLOW:
            print("------")
            print(d[DEBUG_SPECIFIC_ELEMENT])
            print(s1_ev[DEBUG_SPECIFIC_ELEMENT])
            print(s2_ev[DEBUG_SPECIFIC_ELEMENT])
        if not is_normalized:
            return d[:, lenstr1-1, lenstr2-1]
        all_lengths = (~np.ma.getmaskarray(mask_s1_ev) & ~np.ma.getmaskarray(mask_s2_ev)).sum(axis=1)
        return 1 - d[:, lenstr1, lenstr2] / all_lengths


# TODO: Should be a class with default behavior, if input is just one item.
def num_changes_distance(a, b):

    differences = a != b
    num_differences = differences.sum(axis=-1)
    # total_differences_in_sequence = num_differences.sum(axis=-1)
    # return total_differences_in_sequence
    return num_differences


def stack_data(a):
    a_evs, a_fts = zip(*a)
    a_evs_stacked, a_fts_stacked = tf.concat(list(a_evs), axis=0), tf.concat(list(a_fts), axis=0)
    return a_evs_stacked.numpy(), a_fts_stacked.numpy()


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

    if DEBUG_SAME_CASE:
        a, b = pickle.load(io.open("tmp.pkl", mode="rb"))
    loss_singular = DamerauLevenshstein(reader.vocab_len, num_changes_distance)
    distances_singular = []
    sanity_ds_singular = []
    cnt = 0
    for a_i, b_i in zip(a, b):
        if cnt == DEBUG_SPECIFIC_ELEMENT:
            DEBUG_SLOW = True
        a_i = a_i[0][0].numpy().astype(int), a_i[1][0].numpy()
        b_i = b_i[0][0].numpy().astype(int), b_i[1][0].numpy()
        mask_cond = ~((a_i[0] != 0) | (b_i[0] != 0))
        # ft_len = a_i[1].shape[-1]
        # a_i_ev = np.ma.masked_where(mask_cond, a_i[0])
        # b_i_ev = np.ma.masked_where(mask_cond, b_i[0])
        # a_i_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(a_i_ev)[..., None], ft_len, -1), a_i[1])
        # b_i_ft = np.ma.masked_where(np.repeat(np.ma.getmaskarray(b_i_ev)[..., None], ft_len, -1), b_i[1])
        # a_i = a_i_ev, a_i_ft
        # b_i = b_i_ev, b_i_ft
        a_i = a_i[0][~mask_cond], a_i[1][~mask_cond]
        b_i = b_i[0][~mask_cond], b_i[1][~mask_cond]

        r_my = loss_singular(a_i, b_i)
        r_sanity = levenshtein(a_i, b_i)

        DEBUG_SLOW = False
        distances_singular.append(r_my)
        sanity_ds_singular.append(r_sanity)
        cnt += 1
        
    DEBUG_SLOW = True
    loss = DamerauLevenshsteinParallel(reader.vocab_len, reader.max_len, num_changes_distance)
    a_stacked = stack_data(a)
    b_stacked = stack_data(b)
    bulk_distances = loss(a_stacked, b_stacked)
    all_results = np.array([distances_singular, sanity_ds_singular, bulk_distances])
    print(f"All results\n{all_results}")
    if all_results.sum() == 0:
        print("Hmm...")

    if DEBUG_SAME_CASE:
        pickle.dump((a, b), io.open("tmp.pkl", mode="wb"))
