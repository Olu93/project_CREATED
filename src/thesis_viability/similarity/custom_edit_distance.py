


from typing import Any, Callable
import numpy as np
from thesis_readers import MockReader as Reader
from thesis_generators.helper.wrapper import GenerativeDataset
from thesis_commons.modes import DatasetModes, GeneratorModes
from thesis_commons.modes import TaskModes
from scipy.spatial import distance

class DamerauLevenshstein():
    def __init__(self, vocab_len:int, distance_func: Callable):
        self.dist = distance_func
        self.vocab_len = vocab_len
        
    
    def __call__(self, a, b) -> Any:
        
        da = np.zeros(self.vocab_len)
        d = np.zeros((len(a)+3, len(b)+3))
        a_empty = np.zeros_like(a[0])
        b_empty = np.zeros_like(b[0])
        # maxdist = len(a) + len(b)
        maxdist = 9999
        std_cost = 100
        for i in range(1,len(a)):
            d[i, 0] = maxdist
            d[i, 1] = self.dist(a[i], b_empty)
        for j in range(1,len(b)):
            d[0, j] = maxdist
            d[1, j] = self.dist(a_empty, b[j])
        for i in range(2, len(a)):
            db = 0
            for j in range(2, len(b)):
                k = da[b[j]]
                l = db
                if a[i] == b[j]:
                    cost = 0
                    db = self.dist(a_empty, b[j])
                else:
                    cost = self.dist(a[i], b[j])
                    
                tmp = [
                    d[i-1, j-1] + cost,  #substitution
                    d[i,   j-1] + cost,     #insertion
                    d[i-1, j  ] + cost,     #deletion
                    d[k-1, l-1] + (i-k-1) + 1 + (j-l-1)                    
                ]    
                d[i, j] = min(tmp)
            da[a[i]] = i*self.dist(a[i], b_empty)
        return d[len(a), len(b)]
    
def cosine_distance(a, b):
    return distance.cosine(a, b)

if __name__ == "__main__":
    task_mode = TaskModes.NEXT_EVENT_EXTENSIVE
    epochs = 50
    reader = None
    reader = Reader(mode=task_mode).init_meta()
    generative_reader = GenerativeDataset(reader)
    train_data = generative_reader.get_dataset(16, DatasetModes.TRAIN, gen_mode=GeneratorModes.HYBRID, flipped_target=True)
    d_iter = iter(train_data)
    _, y_1 = next(d_iter)
    _, y_2 = next(d_iter)
    loss = DamerauLevenshstein(generative_reader.vocab_len, cosine_distance)
    result = loss(y_1, y_1)
    print(f"Result is {result}")
    
