import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(f"Registered {physical_devices} as physical devices")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)