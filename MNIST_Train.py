import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

MNIST_Classifier = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])
MNIST_Classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)


MNIST_Classifier.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

MNIST_Classifier.save("MNIST_Model")
