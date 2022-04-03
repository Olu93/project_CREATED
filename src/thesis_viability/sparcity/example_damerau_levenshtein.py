from typing import Any, Callable
from unicodedata import is_normalized
import numpy as np
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance


def levenshtein2(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    # print("START")
    # tmp=to_matrix(d, lenstr1, lenstr2)
    # print(tmp[:-1, :-1])
    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
            # print("------")
            # tmp=to_matrix(d, lenstr1, lenstr2)
            # print(tmp[:-1, :-1])
    return d[lenstr1 - 1, lenstr2 - 1]


def to_matrix(d, lenstr1, lenstr2):
    d_tmp = np.zeros((lenstr1 + 1, lenstr2 + 1))
    for (k, l), val in d.items():
        d_tmp[k, l] = val
    return d_tmp


def levenshtein(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    matrix = [list(range(l1 + 1))] * (l2 + 1)
    for zz in list(range(l2 + 1)):
        matrix[zz] = list(range(zz, zz + l1 + 1))
    for zz in list(range(0, l2)):
        for sz in list(range(0, l1)):
            if s1[sz] == s2[zz]:
                matrix[zz + 1][sz + 1] = min(matrix[zz + 1][sz] + 1, matrix[zz][sz + 1] + 1, matrix[zz][sz])
            else:
                matrix[zz + 1][sz + 1] = min(matrix[zz + 1][sz] + 1, matrix[zz][sz + 1] + 1, matrix[zz][sz] + 1)
    distance = float(matrix[l2][l1])
    result = 1.0 - distance / max(l1, l2)
    return result


def levenshtein_wiki(self, a, b) -> Any:
    da = np.zeros(self.vocab_len, dtype=int)
    len_a = len(a)
    len_b = len(b)
    d = np.zeros((len_a + 1, len_b + 1))
    maxdist = len_a + len_b

    d[0, 0] = maxdist
    for i in range(1, len_a + 1):
        d[i, 0] = maxdist
        d[i, 1] = i
    for j in range(1, len_b + 1):
        d[0, j] = maxdist
        d[1, j] = j
    for i in range(1, len_a + 1):
        db = 0
        for j in range(1, len_b + 1):
            k = da[b[j - 1]]
            l = db
            if a[i - 1] == b[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1

            tmp = [
                d[i - 1, j - 1] + cost,  #substitution
                d[i, j - 1] + 1,  #insertion
                d[i - 1, j] + 1,  #deletion
                d[k - 1, l - 1] + (i - k - 1) + 1 + (j - l - 1)
            ]
            d[i, j] = min(tmp)
        da[a[i - 1]] = i
    return d[-1, -1]


class DamerauLevenshstein():

    def __init__(self, vocab_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len

    def __call__(self, s1, s2, is_normalized=False):
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        d = np.zeros((lenstr1 + 1, lenstr2 + 1))
        max_dist = lenstr1 + lenstr2
        d[:, 0] = np.arange(0, lenstr1 + 1)
        d[0, :] = np.arange(0, lenstr2 + 1)
        # d[0,:] = start_val
        # d[:,0] = start_val
        # print("START")
        # tmp=to_matrix(d, lenstr1, lenstr2)
        # print(d)
        for i in range(1, lenstr1 + 1):
            for j in range(1, lenstr2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                d[i, j] = min(
                    d[i - 1, j] + 1,  # deletion
                    d[i, j - 1] + 1,  # insertion
                    d[i - 1, j - 1] + cost,  # substitution
                )
                if i > 1 and j > 1:
                    if i and j and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                        d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
                # print("------")
                # print(d)

        return d[lenstr1, lenstr2] if not is_normalized else 1 - d[lenstr1, lenstr2] / max(lenstr1, lenstr2)


def compute_all(a, b):
    print("===========================")
    print(a)
    print(b)
    result_true = levenshtein2(list(map(str, a)), list(map(str, b)))
    result_normalised_true = levenshtein(list(map(str, a)), list(map(str, b)))
    loss = DamerauLevenshstein(12, cosine_distance)

    computed_result = loss(a, b, is_normalized=False)
    print(f"Correct is {result_true:.5f} -- Count")
    print(f"Result is {computed_result}")
    assert result_true == computed_result, f"Function is not correct anymore {result_true} != {computed_result}"

    computed_result = loss(a, b, is_normalized=True)
    print(f"Correct is {result_normalised_true:.5f}")
    print(f"Result is {computed_result} -- Normed")
    assert result_normalised_true == computed_result, f"Function is not correct anymore {result_normalised_true} != {computed_result}"


class DamerauLevenshsteinParallel():

    def __init__(self, vocab_len: int, max_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len
        self.max_len = max_len

    def __call__(self, s1, s2, is_normalized=False):
        lenstr1 = self.max_len
        lenstr2 = self.max_len
        num_instances = len(s1)
        d = np.zeros((num_instances, lenstr1 + 1, lenstr2 + 1))
        max_dist = lenstr1 + lenstr2
        for i in range(self.max_len+1):
            d[:, :, i] = i
            d[:, i, :] = i

        for i in range(1, lenstr1 + 1):
            for j in range(1, lenstr2 + 1):
                cost = (s1[:, i - 1] != s2[:, j - 1]) * 1
                deletion = d[:, i - 1, j] + 1
                insertion = d[:, i, j - 1] + 1
                substitution = d[:, i - 1, j - 1] + cost
                transposition = np.ones_like(d[:, i, j]) * np.inf
                if i > 1 and j > 1:
                    one_way = s1[:, i - 1] == s2[:, j - 2]
                    bck_way = s1[:, i - 2] == s2[:, j - 1]
                    is_transposed = one_way & bck_way
                    prev_d = d[:, i - 2, j - 2]
                    prev_d[~is_transposed] = np.inf
                    transposition = prev_d + 1
                cases = np.array([
                    deletion,
                    insertion,
                    substitution,
                    transposition,
                ])

                min_d = np.min(cases, axis=0)
                d[:, i, j] = min_d

        return d[:, lenstr1, lenstr2] if not is_normalized else 1 - d[:, lenstr1, lenstr2] / self.max_len


def cosine_distance(a, b):
    return distance.cosine(a, b)


if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(3, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)

    a, b = [0, 8, 3, 4, 5, 6], [7, 8, 9, 5, 4, 11, 4]
    compute_all(a, b)
    a, b = [0, 8, 3, 4, 5, 6], [7, 8, 9, 0, 10, 11, 4]
    compute_all(a, b)
    a, b = [8, 1, 1, 2, 6, 9], [8, 1, 2, 1, 2, 3, 5, 9]
    compute_all(a, b)

    d_iter = iter(train_data)
    a = next(d_iter)[1]
    b = next(d_iter)[1]
    a, b = a[0][0].numpy().astype(int), b[0][0].numpy().astype(int)
    a, b = a[a != 0], b[b != 0]
    compute_all(a, b)

    d_iter = iter(train_data)
    a = next(d_iter)[0][0].numpy()
    b = next(d_iter)[0][0].numpy()
    loss_singular = DamerauLevenshstein(reader.vocab_len, cosine_distance)
    for a_i, b_i in zip(a, b):
        result = loss_singular(a_i, b_i)
        print(f"EDIT DISTANCE d({a_i},{b_i})")
        print(result)
        print(f"Sanity Check: {levenshtein2(list(map(str, a_i)), list(map(str, b_i)))}")
    
    loss = DamerauLevenshsteinParallel(reader.vocab_len, reader.max_len, cosine_distance)
    print(loss(a, b))
