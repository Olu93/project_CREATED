from typing import Any, Callable
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
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def levenshtein(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    matrix = [list(range(l1 + 1))] * (l2 + 1)
    for zz in list(range(l2 + 1)):
      matrix[zz] = list(range(zz,zz + l1 + 1))
    for zz in list(range(0,l2)):
      for sz in list(range(0,l1)):
        if s1[sz] == s2[zz]:
          matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
        else:
          matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
    distance = float(matrix[l2][l1])
    result = 1.0-distance/max(l1,l2)
    return result


class DamerauLevenshstein():

    def __init__(self, vocab_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len

    def __call__(self, a, b) -> Any:
        da = np.zeros(self.vocab_len, dtype=int)
        len_a = len(a)
        len_b = len(b)
        d = np.zeros((len_a+1, len_b+1))
        maxdist = len_a + len_b

        d[0, 0] = maxdist
        for i in range(1, len_a+1):
            d[i, 0] = maxdist
            d[i, 1] = i
        for j in range(1, len_b+1):
            d[0, j] = maxdist
            d[1, j] = j
        for i in range(1, len_a+1):
            db = 0
            for j in range(1, len_b+1):
                k = da[b[j-1]]
                l = db
                if a[i-1] == b[j-1]:
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
            da[a[i-1]] = i
        return d[-1, -1]


def cosine_distance(a, b):
    return distance.cosine(a, b)


if __name__ == "__main__":
    # task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    # epochs = 50
    # reader = None
    # reader = Reader(mode=task_mode).init_meta()
    # generative_reader = GenerativeDataset(reader)
    # train_data = generative_reader.get_dataset(3, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)
    # d_iter = iter(train_data)
    # a = next(d_iter)[1]
    # b = next(d_iter)[1]
    # a, b = a[0][0].numpy().astype(int), b[0][0].numpy().astype(int)
    # a, b = a[a != 0], b[b != 0]
    # a, b = [8,1,1,2,6,9], [8,1,2,1,2,3,5,9]
    a, b = [0, 2, 3, 4, 5, 6], [7, 8, 9, 4, 10, 11, 4]
    print(a)
    print(b)
    result = levenshtein(list(map(str, a)), list(map(str, b)))
    print(f"Correct  is {result:.5f}")
    result = levenshtein2(list(map(str, a)), list(map(str, b)))
    print(f"Correct2 is {result:.5f}")
    loss = DamerauLevenshstein(12, cosine_distance)
    result = loss(a, b)
    print(f"Result is {result:.5f} - It shoulb be >0")
    result = loss(a, a)
    print(f"Result is {result:.5f} - It should be =0")
