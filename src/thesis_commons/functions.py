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

# @tf.function
def sample(inputs):
    mean, logvar = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
    # TODO: Maybe remove the 0.5 and include proper log handling -- EVERYWHERE
    return mean + tf.exp(0.5 * logvar) * epsilon