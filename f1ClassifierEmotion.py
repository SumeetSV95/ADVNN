"""
credits:https://www.kaggle.com/joshithareddy/emotion-recognition-with-vgg16
"""

import numpy as np

import matplotlib.pyplot as plt

import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2,

                                   rotation_range=5,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   # zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255
                                  )

train_dataset = train_datagen.flow_from_directory(directory='../input/fer2013/train',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  subset='training',
                                                  batch_size=64)

valid_dataset = valid_datagen.flow_from_directory(directory='../input/fer2013/train',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  batch_size=64)

test_dataset = test_datagen.flow_from_directory(directory='../input/fer2013/test',
                                                target_size=(48, 48),
                                                class_mode='categorical',
                                                batch_size=64)

base_model = tf.keras.applications.VGG16(input_shape=(48, 48, 3), include_top=False, weights="imagenet")

# Freezing Layers

for layer in base_model.layers[:-4]:
    layer.trainable = False
# Building Model

model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(7, activation='softmax'))

# Model Summary

model.summary()


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    f1_score,
]

lrd = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.50, min_lr=1e-10)

mcp = ModelCheckpoint('model.h5')

es = EarlyStopping(verbose=1, patience=20)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=METRICS)

history = model.fit(train_dataset, validation_data=valid_dataset, epochs=5, verbose=1, callbacks=[lrd, mcp, es])
model.save("FERModel")
