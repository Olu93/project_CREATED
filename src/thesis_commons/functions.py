from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader

import importlib
import io
import json
import pathlib
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K, losses 
# from numpy.typing import np.ndarray

from benedict import benedict
TFLossSpec = Union[str, str, Dict[str, Any]]


def create_path(pthname: str, pth: pathlib.Path):
    print(f"{pthname}: {pth.absolute()}")
    if not pth.exists():
        print(f"ATTENTION: Create a new path {pth.absolute()}")
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def shift_seq_forward(seq):
    seq = np.roll(seq, 1, -1)
    seq[:, 0] = 0
    return seq


def shift_seq_backward(seq):
    seq = np.roll(seq, -1, -1)
    seq[:, -1] = 0
    return seq


def reverse_sequence(data_container):
    original_data = np.array(data_container)
    flipped_data = np.flip(data_container, axis=-2)
    results = np.zeros_like(original_data)
    results[np.nonzero(original_data.sum(-1) != 0)] = flipped_data[(flipped_data.sum(-1) != 0) == True]
    return results


def reverse_sequence_2(data_container, pad_value=0):
    original_data = np.array(data_container)
    results = np.zeros_like(original_data)
    if len(data_container.shape) == 2:
        flipped_data = np.flip(data_container, axis=-1)
        results[np.nonzero(flipped_data != pad_value)] = original_data[(original_data != pad_value)]
    if len(data_container.shape) == 3:
        flipped_data = np.flip(data_container, axis=-2)
        results[np.nonzero(flipped_data.sum(-1) != pad_value)] = original_data[(original_data.sum(-1) != pad_value)]

    return results

def reverse_sequence_3(data_container):
    original_data = data_container.copy()
    equal_rows = np.all(original_data[:] != original_data[:][0], axis=-1)
    results = np.zeros_like(original_data)
    if len(data_container.shape) == 2:
        flipped_data = np.flip(data_container, axis=-1)
        results[np.nonzero(flipped_data != pad_value)] = original_data[(original_data != pad_value)]
    if len(data_container.shape) == 3:
        flipped_data = np.flip(data_container, axis=-2)
        results[np.nonzero(flipped_data.sum(-1) != pad_value)] = original_data[(original_data.sum(-1) != pad_value)]

    return results


def split_params(input):
    mus, logsigmas = input[:, :, 0], input[:, :, 1]
    return mus, logsigmas


# @tf.function
def sample(inputs):
    mean, logvar = inputs
    epsilon = K.random_normal(shape=tf.shape(mean))
    # TODO: Maybe remove the 0.5 and include proper log handling -- EVERYWHERE
    return mean + tf.exp(0.5 * logvar) * epsilon


def stack_data(a):
    a_evs, a_fts = zip(*a)
    a_evs_stacked, a_fts_stacked = tf.concat(list(a_evs), axis=0), tf.concat(list(a_fts), axis=0)
    return a_evs_stacked.numpy(), a_fts_stacked.numpy()


# def reverse_sequence(data_container):
#     original_data = tf.TensorArray(data_container)
#     flipped_data = np.flip(data_container, axis=-2)
#     results = np.zeros_like(original_data)
#     results[np.nonzero(original_data.sum(-1) != 0)] = flipped_data[(flipped_data.sum(-1) != 0) == True]
#     return results


def extract_loss(fn: losses.Loss) -> TFLossSpec:
    result = {}
    result['module_name'] = fn.__module__
    result['class_name'] = fn.__class__.__name__
    result['config'] = fn.get_config()
    return result


def instantiate_loss(cls_details: TFLossSpec) -> object:
    module = importlib.import_module(cls_details.get('module_name'))
    class_description = getattr(module, cls_details.get('class_name'))
    cfg = cls_details.get('config')
    # fns = cfg.pop('fns')
    instance = class_description().from_config(cfg)
    return instance


def save_loss(path: pathlib.Path, fn:losses.Loss):
    cls_details = extract_loss(fn)
    try:
        new_path = path / "loss.json"
        json.dump(cls_details, io.open(new_path, 'w'))
        return new_path.absolute()
    except Exception as e:
        print(e)
    return None


def save_metrics(path: pathlib.Path, fns: Sequence[losses.Loss]):
    cls_details = [extract_loss(fn) for fn in fns]
    paths = []
    try:
        for i in enumerate(cls_details):
            new_path = path / f"metrics_{i}.json"
            json.dump(cls_details, io.open(new_path, 'w'))
            paths.append(path.absolute(new_path))
        return paths
    except Exception as e:
        print(e)
    return None


def load_loss(path: pathlib.Path):
    try:
        cls_details = json.load(io.open(path, 'r'))
        instance = instantiate_loss(cls_details)
        return instance
    except Exception as e:
        print(e)
    return None


def remove_padding(data:Sequence[Sequence[int]], pad_id:int=0) -> Sequence[Sequence[int]]:
    result:Sequence[Sequence[int]] = []
    for row in data:
        indices = [idx for idx, elem in enumerate(row) if elem != pad_id]
        start = min(indices) if len(indices) else 0
        end = max(indices)+1  if len(indices) else len(row)
        subset = row[start:end]
        result.append(subset)
    return result

def merge_dicts(*args, **kwargs) -> benedict:
    d = benedict()
    for addition in args:
        d.merge(addition, **kwargs)
    return d
    
        
        
def decode_sequences(data:Sequence[Sequence], idx2vocab:Dict[int, str] = None) -> Sequence[str]:
    return [" > ".join([f"{i:02d}" for i in row]) for row in data]

def decode_sequences_str(data:Sequence[Sequence], idx2vocab:Dict[int, str]) -> Sequence[str]:
    return [" > ".join([str(idx2vocab[i]) for i in row]) for row in data]


# https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
def pdf(x, mean, cov):
    return np.exp(logpdf(x, mean, cov))


def logpdf(x, mean, cov):
    # `eigh` assumes the matrix is Hermitian.
    vals, vecs = np.linalg.eigh(cov)
    logdet = np.sum(np.log(vals))
    valsinv = np.array([1. / v for v in vals])
    # `vecs` is R times D while `vals` is a R-vector where R is the matrix
    # rank. The asterisk performs element-wise multiplication.
    U = vecs * np.sqrt(valsinv)
    rank = len(vals)
    dev = x - mean
    # "maha" for "Mahalanobis distance".
    maha = np.square(np.dot(dev, U)).sum()
    log2pi = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)

def extract_padding_mask(a: np.ndarray) -> np.ndarray:
    m1:np.ndarray = a==0
    m2:np.ndarray = (~m1).cumsum(-1)>0 # cumprod solution would also be possible
    return m2

def extract_padding_end_indices(a:np.ndarray) -> np.ndarray:
    m1:np.ndarray = a==0
    m2:np.ndarray = (~m1).cumsum(-1)>0 # cumprod solution would also be possible
    
    m = (~m1).cumsum(-1)
    m[m==0]=9999
    starts = m.argmin(-1)-1
    return starts

# def extract_padding_start_end_indices(a:np.ndarray) -> np.ndarray:
#     m1:np.ndarray = a==0
#     m2:np.ndarray = (~m1).cumprod(-1)>0
#     mask:np.ndarray = m1 & m2
#     start = np.where(mask.any(1), mask.argmax(1), a.shape[1]-1)
#     end = np.where(mask.any(1), mask.argmin(1), a.shape[1]-1)
#     return start, end



# https://gist.github.com/righthandabacus/f1d71945a49e2b30b0915abbee668513
def sliding_window(events, win_size) -> np.ndarray:
    '''Slding window view of a 2D array a using numpy stride tricks.
    For a given input array `a` and the output array `b`, we will have
    `b[i] = a[i:i+w]`
    
    Args:
        a: numpy array of shape (N,M)
    Returns:
        numpy array of shape (K,w,M) where K=N-w+1
    '''
    return np.lib.stride_tricks.sliding_window_view(events, (1, win_size))