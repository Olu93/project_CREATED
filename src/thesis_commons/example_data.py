import tensorflow as tf
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader

from thesis_commons.functions import shift_seq_backward


class RandomExample():

    def __init__(self, vocab_len, max_len) -> None:
        self.vocab_len = vocab_len
        self.max_len = max_len

    def generate_token_only_generator_example(self, batch_size=42):
        current = tf.minimum(tf.cast(tf.random.gamma(shape=(
            batch_size,
            self.max_len,
        ), alpha=0.5, beta=2) * self.vocab_len, tf.int32), self.vocab_len-1)
        prev = shift_seq_backward(current)
        prev_prev = shift_seq_backward(prev)
        result = tf.stack([prev, prev_prev], axis=1)
        return result, current
