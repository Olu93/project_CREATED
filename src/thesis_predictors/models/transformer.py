import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Dropout, Embedding, Multiply, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.optimizer_v2.adam import Adam

from thesis_readers.readers.MockReader import MockReader

from ..tests.example_inputs import TestInput
from .model_commons import ModelInterface

from thesis_readers.helper.modes import TaskModeType


class Seq2SeqTransformerModelOneWay(ModelInterface):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, vocab_len, max_len, feature_len, input_type=0, embed_dim=10, ff_dim=10, pos_embed_dim=10, num_heads=3, rate1=0.1, rate2=0.1, *args, **kwargs):
        super(Seq2SeqTransformerModelOneWay, self).__init__(vocab_len, max_len, feature_len, input_type=input_type, *args, **kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_len, output_dim=embed_dim, mask_zero=0)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=pos_embed_dim, mask_zero=0)
        # Dimensions of token embeddings, position embeddings and feature length
        # self.pos_input = layers.Lambda(self.concat_with_position)
        self.attention_dim = embed_dim + pos_embed_dim if input_type == 0 else embed_dim + pos_embed_dim + feature_len - 1
        self.transformer_block = TransformerBlock(self.attention_dim, num_heads, ff_dim, rate1)
        # self.avg_pooling_layer = layers.GlobalAveragePooling1D()
        self.dropout1 = Dropout(rate2)
        # self.dense = Dense(20, activation='relu')
        # self.dropout2 = Dropout(rate2)
        self.output_layer = TimeDistributed(Dense(self.vocab_len, activation='softmax'))

    def call(self, inputs):
        # x = Event indices
        x, features = inputs if self.input_type != 0 else (inputs[0], None)
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_emb(positions)
        positions = tf.ones_like(x[..., None]) * positions
        x = self.token_emb(x)
        if features is not None:
            x = tf.concat([x, features], axis=-1)
        # positions = self.pos_input(x, positions)
        # positions = self.concat_with_position(x, positions)
        # x = x + positions
        x = tf.concat([x, positions], axis=-1)
        x = self.transformer_block(x)

        # x = self.avg_pooling_layer(x)
        x = self.dropout1(x)
        # x = self.dense(x)
        # x = self.dropout2(x)
        y_pred = self.output_layer(x)

        return y_pred

    @tf.function
    def concat_with_position(self, x, positions):
        return tf.concat([x, positions], axis=-1)


class Seq2SeqTransformerModelOneWaySeperated(Seq2SeqTransformerModelOneWay):
    task_mode_type = TaskModeType.FIX2FIX

    def __init__(self, *args, **kwargs):
        super(Seq2SeqTransformerModelOneWaySeperated, self).__init__(*args, input_type=1, **kwargs)
        # super(Seq2SeqTransformerModelOneWay, self).__init__(*args, **kwargs)
        # super(DualInput, self).__init__()


# ==========================================================================================
class TransformerModelOneWaySimple(Seq2SeqTransformerModelOneWay):
    task_mode_type = TaskModeType.FIX2ONE

    def __init__(self, vocab_len, max_len, *args, **kwargs):
        super(TransformerModelOneWaySimple, self).__init__(vocab_len, max_len, *args, **kwargs)
        self.output_layer = Dense(vocab_len)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        # x = self.avg_pooling_layer(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        y_pred = self.activation_layer(x)

        return y_pred


class TransformerModelTwoWay(Seq2SeqTransformerModelOneWay):
    def __init__(self, vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1) -> None:
        super(TransformerModelTwoWay, self).__init__(vocab_len, max_len, embed_dim=10, ff_dim=10, num_heads=3, rate1=0.1, rate2=0.1)
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_len, embed_dim)
        self.embedding_reverse = TokenAndPositionEmbedding(max_len, vocab_len, embed_dim)
        self.reverse = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reverse(x, axes=-1), output_shape=(max_len, ))
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        inputs = inputs[0]
        x = inputs
        x_reverse = self.reverse(inputs)
        x = self.embedding(x)
        x_reverse = self.embedding_reverse(x_reverse)
        x = self.transformer_block(x)
        x_reverse = self.transformer_block(x_reverse)
        x = self.concat([x, x_reverse])
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.time_distributed_layer(x)
        y_pred = self.activation_layer(x)

        return y_pred


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        inputs = inputs
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=0)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=0)
        self.zero = tf.constant(0, dtype=tf.float32)
        self.multiply = Multiply()

    def call(self, x):
        maxlen = self.maxlen
        positions = tf.range(start=0, limit=maxlen, delta=1, dtype=tf.float32)
        # zero_indices = tf.cast(tf.not_equal(x, self.zero), tf.float32)
        # positions = self.multiply([positions, zero_indices])
        positions = self.pos_emb(tf.cast(positions, tf.int32))
        x = self.token_emb(x)
        return (x + positions)


if __name__ == "__main__":
    reader = MockReader().init_log().init_data()
    data = reader.get_dataset()
    epochs = 1
    adam_init = 0.001
    print("Transformer Bi:")
    model = Seq2SeqTransformerModelOneWay(reader.vocab_len, reader.max_len, reader.feature_len)
    model.compile(loss=model.loss_fn, optimizer=Adam(adam_init), metrics=model.metrics)
    # model.summary()

    prediction = model(data)
    print(prediction)
