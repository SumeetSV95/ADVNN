import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np

MNIST_Classifier = keras.Sequential([
tf.keras.layers.Dense(units=756,use_bias=True,activation='relu'),
tf.keras.layers.Dense(units=100,use_bias=True,activation='relu'),
tf.keras.layers.Dense(units=10,use_bias=True)
])
MNIST_Classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
x_train = x_train.astype(np.float)/255
x_test = x_test.astype(np.float)/255

MNIST_Classifier.fit(x=x_train,y=y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test))

MNIST_Classifier.save("MNIST_Model")
