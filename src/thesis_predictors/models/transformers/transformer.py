import tensorflow as tf

keras = tf.keras
from keras import backend as K, losses, metrics, utils, layers, optimizers, models

from thesis_commons.modes import FeatureModes, TaskModeType
from thesis_readers.readers.MockReader import MockReader

from ...model_commons import HybridInput, ModelInterface, TokenInput, VectorInput


class Transformer(ModelInterface):
    def __init__(self, embed_dim, ff_dim, pos_embed_dim, num_heads, rate1, rate2, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads
        self.rate1 = rate1
        self.rate2 = rate2
        # self.pos_input_placeholder = layers.Input((self.max_len, 1))
        self.pos_embedder = layers.Embedding(input_dim=self.max_len, output_dim=self.pos_embed_dim, mask_zero=0)
        self.token_embedder = layers.Embedding(input_dim=self.vocab_len, output_dim=embed_dim, mask_zero=0)
        self.transformer_block = None
        self.dropout1 = layers.Dropout(rate2)
        self.output_layer = layers.TimeDistributed(layers.Dense(self.vocab_len, activation='softmax'))


class Seq2SeqTransformerModelOneWay(Transformer):
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = TokenInput()

    def __init__(self, embed_dim=10, ff_dim=10, pos_embed_dim=10, num_heads=3, rate1=0.1, rate2=0.1, **kwargs):
        super(Seq2SeqTransformerModelOneWay, self).__init__(embed_dim=embed_dim,
                                                            ff_dim=ff_dim,
                                                            pos_embed_dim=pos_embed_dim,
                                                            num_heads=num_heads,
                                                            rate1=rate1,
                                                            rate2=rate2,
                                                            **kwargs)
        # Dimensions of token embeddings, position embeddings and feature length
        self.transformer_block = TransformerBlock(embed_dim + pos_embed_dim, num_heads, ff_dim, rate1)

    def call(self, inputs):
        # TODO: Impl: all types of inputs
        x = inputs
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_embedder(positions)
        positions = tf.ones_like(x[..., None]) * positions
        x = self.token_embedder(x)
        x = tf.concat([x, positions], axis=-1)
        x = self.transformer_block(x)
        x = self.dropout1(x)
        y_pred = self.output_layer(x)

        return y_pred


class Seq2SeqTransformerModelOneWaySeperated(Transformer):
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = HybridInput()

    def __init__(self, embed_dim=10, ff_dim=10, pos_embed_dim=10, num_heads=3, rate1=0.1, rate2=0.1, **kwargs):
        super(Seq2SeqTransformerModelOneWaySeperated, self).__init__(embed_dim=embed_dim,
                                                                     ff_dim=ff_dim,
                                                                     pos_embed_dim=pos_embed_dim,
                                                                     num_heads=num_heads,
                                                                     rate1=rate1,
                                                                     rate2=rate2,
                                                                     **kwargs)
        self.transformer_block = TransformerBlock(self.embed_dim + self.pos_embed_dim + self.feature_len, self.num_heads, self.ff_dim, self.rate1)

    def call(self, inputs):
        # TODO: Impl: all types of inputs
        x, features = inputs
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_embedder(positions)
        positions = tf.ones_like(x[..., None]) * positions
        x = self.token_embedder(x)
        x = tf.concat([x, features], axis=-1)
        x = tf.concat([x, positions], axis=-1)
        x = self.transformer_block(x)

        x = self.dropout1(x)
        y_pred = self.output_layer(x)

        return y_pred


class Seq2SeqTransformerModelOneWayFull(Transformer):
    task_mode_type = TaskModeType.FIX2FIX
    input_interface = VectorInput()

    def __init__(self, embed_dim=10, ff_dim=10, pos_embed_dim=10, num_heads=3, rate1=0.1, rate2=0.1, **kwargs):
        super(Seq2SeqTransformerModelOneWayFull, self).__init__(embed_dim=embed_dim,
                                                                ff_dim=ff_dim,
                                                                pos_embed_dim=pos_embed_dim,
                                                                num_heads=num_heads,
                                                                rate1=rate1,
                                                                rate2=rate2,
                                                                **kwargs)
        self.transformer_block = TransformerBlock(self.pos_embed_dim + self.feature_len, self.num_heads, self.ff_dim, self.rate1)

    def call(self, inputs):
        # TODO: Impl: all types of inputs
        x = inputs
        # pos_input_placeholder = layers.Input((self.max_len, 1))
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_embedder(positions)
        positions = tf.ones_like(tf.reduce_mean(x, -1, keepdims=True)) * positions
        x = tf.concat([x, positions], axis=-1)
        x = self.transformer_block(x)

        x = self.dropout1(x)
        y_pred = self.output_layer(x)

        return y_pred


# ==========================================================================================
class TransformerModelOneWaySimple(Seq2SeqTransformerModelOneWay):
    task_mode_type = TaskModeType.FIX2ONE

    def __init__(self, vocab_len, max_len, *args, **kwargs):
        super(TransformerModelOneWaySimple, self).__init__(vocab_len, max_len, *args, **kwargs)
        self.output_layer = layers.Dense(vocab_len)
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
        self.reverse = layers.Lambda(lambda x: K.reverse(x, axes=-1), output_shape=(max_len, ))
        self.concat = layers.Concatenate()

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
        self.ffn = models.Sequential([
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
        self.multiply = layers.Multiply()

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
    epochs = 1
    adam_init = 0.001

    print("Transformer Mono:")
    data = reader.get_dataset(ft_mode=FeatureModes.EVENT)
    model = Seq2SeqTransformerModelOneWay(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len)
    model.compile(loss=model.loss_fn, optimizer=optimizers.Adam(adam_init), metrics=model.metrics)
    model.summary()
    print("Transformer Bi:")
    data = reader.get_dataset(ft_mode=FeatureModes.FULL)
    model = Seq2SeqTransformerModelOneWaySeperated(vocab_len=reader.vocab_len, max_len=reader.max_len, feature_len=reader.feature_len)
    model.compile(loss=model.loss_fn, optimizer=optimizers.Adam(adam_init), metrics=model.metrics)
    model.summary()
    prediction = model.fit(data)

    example = next(iter(data))
    print(model.predict(data))
