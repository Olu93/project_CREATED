import pathlib
import numpy as np
import tensorflow as tf

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

def reverse_sequence_2(data_container):
    original_data = np.array(data_container)
    results = np.zeros_like(original_data)
    if len(data_container.shape) == 2:  
        flipped_data = np.flip(data_container, axis=-1)
        results[np.nonzero(flipped_data != 0)] = original_data[(original_data != 0)]
    if len(data_container.shape) == 3:  
        flipped_data = np.flip(data_container, axis=-2)
        results[np.nonzero(flipped_data.sum(-1) != 0)] = original_data[(original_data.sum(-1) != 0)]
    
    return results

def split_params(input):
    mus, logsigmas = input[:,:,0], input[:,:,1]
    return mus, logsigmas

# @tf.function
def sample(inputs):
    mean, logvar = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
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