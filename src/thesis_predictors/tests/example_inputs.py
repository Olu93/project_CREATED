import tensorflow as tf

from thesis_commons import modes


class TestInput():
    def __init__(self, ft_mode: modes.FeatureModes = modes.FeatureModes.FULL) -> None:
        self.y_true = tf.constant([[1, 2, 1, 0, 0], [1, 2, 1, 1, 0], [1, 2, 1, 1, 2], [1, 2, 0, 0, 0]], dtype=tf.int32)
        self.sample_weights = tf.random.uniform((self.y_true.shape[0],1))
        self.x_feat = None
        self.feature_len = 42
        if ft_mode == modes.FeatureModes.FULL:
            self.x_feat = tf.random.uniform(shape=self.y_true.shape, minval=1, maxval=10, dtype=tf.int32)
        self.y_pred = tf.constant(
            [
                [
                    [0.04, 0.95, 0.01],
                    [0.10, 0.10, 0.80],
                    [0.20, 0.70, 0.10],
                    [0.20, 0.10, 0.70],
                    [0.70, 0.20, 0.10],
                ],
                [
                    [0.04, 0.01, 0.95],
                    [0.10, 0.10, 0.80],
                    [0.05, 0.93, 0.02],
                    [0.91, 0.04, 0.05],
                    [0.70, 0.20, 0.10],
                ],
                [
                    [0.05, 0.04, 0.91],
                    [0.90, 0.00, 0.10],
                    [0.91, 0.04, 0.05],
                    [0.91, 0.08, 0.01],
                    [0.91, 0.09, 0.00],
                ],
                [
                    [0.05, 0.94, 0.01],
                    [0.10, 0.80, 0.10],
                    [0.05, 0.93, 0.02],
                    [0.00, 0.95, 0.05],
                    [0.02, 0.88, 0.10],
                ],
            ],
            dtype=tf.float32,
        )

    def get_dataset(self, *args, **kwargs):
        return tf.data.Dataset.from_tensor_slices([self.x_feat, self.y_true, self.sample_weights]).batch(1)