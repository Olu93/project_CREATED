
import tensorflow as tf

from thesis_commons import modes
from tensorflow.python.keras import layers, models


class EmbedderLayer(models.Model):
    def __init__(self, feature_len=None, max_len=None, ff_dim=None, vocab_len=None, embed_dim=None, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(EmbedderLayer, self).__init__(*args, **kwargs)
        self.embedder = layers.Embedding(vocab_len, embed_dim, mask_zero=mask_zero, *args, **kwargs)
        self.combiner = layers.Concatenate(axis=-1)
        self.feature_len: int = feature_len
        self.vocab_len: int = vocab_len

    def call(self, inputs, **kwargs):
        inputs = self.prepare_features(inputs, **kwargs)
        x = self.construct_embedding(inputs, **kwargs)
        self.compute_feature_len(x, **kwargs)
        return x

    def prepare_features(self, inputs, **kwargs):
        events, other_features = inputs
        events = tf.cast(events, tf.int32)
        return events, other_features
   
    def construct_embedding(self, inputs, **kwargs):
        events, other_features = inputs
        x = self.embedder(events)
        return x, other_features

    def compute_feature_len(self, x, **kwargs):
        self.feature_len = x.shape[-1]

    def get_config(self):
        return {"feature_len": self.feature_len, "vocab_len": self.vocab_len}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OnehotEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(OnehotEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)
        # self.embedder = layers.CategoryEncoding(vocab_len, output_mode="one_hot")
        # self.test = layers.Lambda(lambda ev_sequence: self.embedder(ev_sequence))
        self.embedder = layers.Lambda(OnehotEmbedderLayer._one_hot, arguments={'num_classes': vocab_len})
        self.concatenator = layers.Concatenate(name="concat_embedding_and_features")

    @classmethod
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, tf.uint8), num_classes=num_classes)

    def construct_embedding(self, inputs, **kwargs):
        x, features = super().construct_embedding(inputs, **kwargs)
        x = self.concatenator([x, features])
        return x


class TokenEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        print(__class__)
        super(TokenEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)
        self.concatenator = layers.Concatenate(name="concat_embedding_and_features")



    def construct_embedding(self, inputs, **kwargs):
        x, features = super().construct_embedding(inputs, **kwargs)
        x = self.concatenator([x, features])
        return x



class HybridEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0, *args, **kwargs) -> None:
        super(HybridEmbedderLayer, self).__init__(vocab_len=vocab_len, embed_dim=embed_dim, mask_zero=mask_zero, *args, **kwargs)
        self.concatenator = layers.Concatenate(name="concat_embedding_and_features")

    def construct_embedding(self, inputs, **kwargs):
        x, features = super().construct_embedding(inputs, **kwargs)
        x = self.concatenator([x, features])

        return x


class VectorEmbedderLayer(EmbedderLayer):
    def __init__(self, vocab_len, embed_dim, mask_zero=0) -> None:
        super(VectorEmbedderLayer, self).__init__(vocab_len, embed_dim, mask_zero)

    def construct_embedding(self, inputs, **kwargs):
        x, features = super().construct_embedding(inputs, **kwargs)

        return features


class EmbedderConstructor():
    def __new__(cls, **kwargs) -> EmbedderLayer:
        ft_mode = kwargs.pop('ft_mode', None)
        input_mode = modes.InputModeType.type(ft_mode)
        if input_mode == modes.InputModeType.TOKEN_INPUT:
            return TokenEmbedderLayer(**kwargs)
        if input_mode == modes.InputModeType.VECTOR_INPUT:
            return VectorEmbedderLayer(**kwargs)
        if input_mode == modes.InputModeType.DUAL_INPUT:
            return HybridEmbedderLayer(**kwargs)
        print(f"Attention! Input mode is not specified -> ft_mode = {ft_mode} | input_mode = {input_mode}")
        return OnehotEmbedderLayer(**kwargs)