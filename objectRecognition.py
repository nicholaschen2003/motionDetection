import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
import random

# def generator(batch_size, data_set_list):
#     index = 0
#     while True:
#         batchX, batchY = [], []
#         for i in range(batch_size):
#             if index >= len(data_set_list):
#                 index = 0
#             batchX.append(data_set_list[index][0].reshape((128,128,3)).astype('int32'))
#             batchY.append(data_set_list[index][1])
#             index += 1
#         yield np.array(batchX), np.array(batchY)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        s = f"Epoch {epoch+1}\n"
        for key in keys:
            s += f"{key}: {logs[key]:.4f} - "
        print(s[:-3], file=open('net/log.txt', 'a'))

callback = tf.keras.callbacks.ModelCheckpoint(filepath="net/5/checkpoints/",
                                                verbose=1,
                                                save_best_only=True,
                                                save_freq='epoch')

TRAIN_EPOCHS = 10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential([
        layers.InputLayer(input_shape = input_shape),
        layers.experimental.preprocessing.Rescaling(1./255),
        # layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # layers.experimental.preprocessing.RandomRotation(0.5),
        # layers.experimental.preprocessing.RandomContrast(0.5),
        # layers.experimental.preprocessing.RandomTranslation(height_factor=0.3, width_factor=0.3),
        # layers.experimental.preprocessing.RandomZoom(height_factor=0.3),
        # layers.experimental.preprocessing.RandomCrop(128,128),
        ]) #use preprocessing to make dataset larger
        #128x128x3
        self.model.add(layers.Conv2D(8, 13, activation = 'relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.2))
        #116x116x8
        self.model.add(layers.AveragePooling2D(pool_size = 2))
        #58x58x8
        self.model.add(layers.Conv2D(16, 11, activation = 'relu'))
        #48x48x16
        self.model.add(layers.AveragePooling2D(pool_size = 2))
        #24x24x16
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation = 'relu'))
        self.model.add(layers.Dense(2, activation = 'softmax'))
        self.optimizer = optimizers.Adam(lr=0.001)
        self.loss = losses.CategoricalCrossentropy()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
        print(summaryStr, file=open('net/log.txt', 'a'))

if __name__ == "__main__":
    random_flip = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
    random_rotation = layers.experimental.preprocessing.RandomRotation(0.5)
    random_contrast = layers.experimental.preprocessing.RandomContrast(0.5)
    random_translation = layers.experimental.preprocessing.RandomTranslation(height_factor=0.3, width_factor=0.3)
    random_zoom = layers.experimental.preprocessing.RandomZoom(height_factor=0.3)
    random_crop = layers.experimental.preprocessing.RandomCrop(128,128)
    train_ds0 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16)
    train_ds1 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_flip(x),y))
    train_ds2 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_rotation(x),y))
    train_ds3 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_contrast(x),y))
    train_ds4 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_translation(x),y))
    train_ds5 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_zoom(x),y))
    train_ds6 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="training", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_crop(x),y))
    val_ds0 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16)
    val_ds1 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_flip(x),y))
    val_ds2 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_rotation(x),y))
    val_ds3 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_contrast(x),y))
    val_ds4 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_translation(x),y))
    val_ds5 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_zoom(x),y))
    val_ds6 = tf.keras.preprocessing.image_dataset_from_directory("datasets", label_mode="categorical", validation_split=(1./11), subset="validation", image_size=(128,128), seed=123, batch_size=16).map(lambda x,y: (random_crop(x),y))
    train_ds = train_ds0.concatenate(train_ds1).concatenate(train_ds2).concatenate(train_ds3).concatenate(train_ds4).concatenate(train_ds5).concatenate(train_ds6).shuffle(1250)
    val_ds = val_ds0.concatenate(val_ds1).concatenate(val_ds2).concatenate(val_ds3).concatenate(val_ds4).concatenate(val_ds5).concatenate(val_ds6)
    load = input("Enter path to model to be loaded, or hit enter for no model: ")
    if load == "":
        net = Net((128, 128, 3))
        print(net)
        net = net.model
    else:
        net = tf.keras.models.load_model(load)
    results = net.fit(x=train_ds,
                        validation_data=val_ds,
                        epochs = TRAIN_EPOCHS,
                        verbose = 1,
                        shuffle = True,
                        callbacks=[callback, CustomCallback()]) #saving and logging
    tf.keras.models.save_model(net, "net/5/final/")
