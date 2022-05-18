# %%
from pathlib import Path
import tensorflow.python.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2
import abc
from tensorflow.python.util.tf_export import keras_export

REDUCTION = ReductionV2


# %%
class CustomLoss(keras.losses.LossFunctionWrapper):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super(CustomLoss, self).__init__(reduction=REDUCTION.NONE, name=name or self.__class__.__name__, **kwargs)
        self.kwargs = kwargs
        self.reduction = reduction
        self.from_logits = False


    def get_config(self):
        cfg = {**self.kwargs, **super().get_config()}
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SpecialLoss(CustomLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super(SpecialLoss, self).__init__(fn=tf.keras.losses.mean_absolute_error, name=name, reduction=reduction, **kwargs)


class SpecialMetric(CustomLoss):
    def __init__(self, reduction=REDUCTION.NONE, name=None, **kwargs):
        super(SpecialMetric, self).__init__(fn=tf.keras.losses.mean_absolute_error, name=name, reduction=reduction, **kwargs)



class TensorflowModelMixin(keras.models.Model):
    def __init__(self, *args, **kwargs) -> None:
        print(__class__)
        super(TensorflowModelMixin, self).__init__(*args, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        optimizer = optimizer or self.optimizer
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_graph(self):
        events = tf.keras.layers.Input(shape=(32, ), name="events")
        features = tf.keras.layers.Input(shape=(32, self.feature_len), name="event_attributes")
        inputs = [events, features]
        summarizer = keras.models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer


# THIS CREATED A WARNING: WARNING:tensorflow:5 out of the last 37 calls to <function Model.make_predict_function.
class InputInterface(abc.ABC):
    @classmethod
    def summary(self):
        raise NotImplementedError()


class HybridInput(InputInterface):
    def summary(self):
        events = tf.keras.layers.Input(shape=(32, ), name="events")
        features = tf.keras.layers.Input(shape=(32, self.feature_len), name="event_attributes")
        inputs = [events, features]
        summarizer = keras.models.Model(inputs=[inputs], outputs=self.call(inputs))
        return summarizer.summary()


class CustomModel(HybridInput, TensorflowModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute = keras.layers.Dense(1)
        self.compute2 = keras.layers.Dense(1)

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = SpecialLoss()
        metrics = [keras.metrics.Accuracy()]
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def call(self, inputs):
        x, z = inputs
        return self.compute(x) + self.compute2(z)

    def train_step(self, data):
        # print("Train-Step")
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) >= 4:
            x, z, y, sample_weight = data
        else:
            sample_weight = None
            x, z, y = data

        with tf.GradientTape() as tape:
            y_pred = self((x, z), training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            # print("y.shape")
            # print(y.shape)
            # print("y_pred.shape")
            # print(y_pred.shape)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # print("Test-Step")
        # Unpack the data
        x, z, y = data
        # Compute predictions
        y_pred = self((x, z), training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        return config


# %%
test_path = Path("./junk/test_model").absolute()
print(f'Save at {test_path}')
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(verbose=2, filepath=test_path, save_best_only=True)

# %%
# Construct and compile an instance of CustomModel
model = CustomModel()
model.compile(optimizer="adam", loss=None, metrics=None, run_eagerly=True)

# You can now use sample_weight argument
x_data = np.random.random((1000, 32))
z_data = np.random.random((1000, 32))
sw = np.random.random((1000, 1))
y = np.random.random((1000, 1))
in_train = tf.data.Dataset.from_tensor_slices((x_data, z_data, y, sw)).batch(5)

x_data = np.random.random((1000, 32))
z_data = np.random.random((1000, 32))
sw = np.random.random((1000, 1))
y = np.random.random((1000, 1))
in_val = tf.data.Dataset.from_tensor_slices((x_data, z_data, y)).batch(5)
# %%
model.fit(in_train, validation_data=in_val, epochs=2, callbacks=[model_checkpoint_callback])
# %%
new_model = keras.models.load_model(test_path)
# new_model = keras.models.load_model(test_path, custom_objects={"SpecialLoss": SpecialLoss(),"SpecialMetric": SpecialMetric()})
# %%
x_data = np.random.random((3, 32))
z_data = np.random.random((3, 32))
new_model.predict((x_data, z_data))[:10]
# %%
import json
import io
import pathlib
json.dump(model.loss.get_config(), io.open(pathlib.Path(test_path)/"loss.json", "w"))
# %%
