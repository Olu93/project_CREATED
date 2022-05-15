# %%
from pathlib import Path
import tensorflow.python.keras as keras
import numpy as np
import tensorflow as tf

# %%
class CustomLoss(keras.losses.Loss):
    def __init__(self) -> None:
        super(CustomLoss, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name="test_loss")
        self.fn = tf.keras.losses.MeanSquaredError()
        
    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred)
        
    def get_config(self):
        return {}
   
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        loss = CustomLoss()
        metrics = [CustomLoss()]
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)
    
    def train_step(self, data):
        # print("Train-Step")
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
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
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

# %% 
test_path = Path("./junk/test_model").absolute()
print(f'Save at {test_path}')
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    verbose=2,
    filepath=test_path,
    save_best_only=True)
    
# %% 
# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss=None, metrics=None, run_eagerly=True)

# You can now use sample_weight argument
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
val_x = np.random.random((1000, 32))
val_y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, validation_data=(val_x, val_y), epochs=3, callbacks=[model_checkpoint_callback])
# %%
new_model = keras.models.load_model(test_path, custom_objects={"CustomLoss": CustomLoss()})
# %%
new_model.predict(x)[:10]