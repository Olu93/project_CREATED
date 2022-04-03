


from typing import Any, Callable
import numpy as np
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance

class DamerauLevenshstein():

    def __init__(self, vocab_len: int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len

    def __call__(self, s1, s2, is_normalized=False):
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        d = np.zeros((lenstr1+1, lenstr2+1))
        max_dist = lenstr1 + lenstr2
        d[:, 0] = np.arange(0, lenstr1+1)
        d[0, :] = np.arange(0, lenstr2+1)

        for i in range(1,lenstr1+1):
            for j in range(1,lenstr2+1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                d[i,j] = min(
                            d[i-1,j] + 1, # deletion
                            d[i,j-1] + 1, # insertion
                            d[i-1,j-1] + cost, # substitution
                            )
                if i>1 and j>1:
                    if i and j and s1[i-1]==s2[j-2] and s1[i-2] == s2[j-1]:
                        d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
                # print("------")
                # print(d)

        return d[lenstr1,lenstr2] if not is_normalized else 1 - d[lenstr1,lenstr2] / max(lenstr1,lenstr2)
    
def cosine_distance(a, b):
    return distance.cosine(a, b)

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(3, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)
    d_iter = iter(train_data)
    _, y_1 = next(d_iter)
    _, y_2 = next(d_iter)
    loss = DamerauLevenshstein(generative_reader.vocab_len, cosine_distance)
    result = loss(y_1, y_1)
    print(f"Result is {result}")
    
