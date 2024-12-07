import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Create a simple model
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(20,))])
model.compile(optimizer='adam', loss='mse')

# Generate random data
x = np.random.rand(100, 20)
y = np.random.rand(100, 10)

# Train the model
model.fit(x, y, epochs=5, verbose=1)
