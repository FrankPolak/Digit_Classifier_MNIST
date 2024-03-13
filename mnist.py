import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Load MNIST data
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# Scaling
train_images = train_images / 255.
test_images = test_images / 255.
scaled_train_images = train_images[..., np.newaxis]
scaled_test_images = test_images[..., np.newaxis]

# Build the CNN model
model = Sequential([
    Conv2D(8, (3, 3), padding='SAME', activation='relu', input_shape=scaled_train_images[0].shape),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(scaled_train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(scaled_test_images, test_labels)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Test loss: 0.21296660602092743
# Test accuracy: 0.9340999722480774