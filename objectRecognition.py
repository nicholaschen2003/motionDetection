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

def generator(batch_size, data_set_list):
    index = 0
    while True:
        batchX, batchY = [], []
        for i in range(batch_size):
            if index >= len(data_set_list[0]):
                index = 0
            batchX.append(data_set_list[0][index].reshape((128,128,3)).astype('int32'))
            batchY.append(data_set_list[1][index])
            index += 1
        yield np.array(batchX), np.array(batchY)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        s = f"Epoch {epoch+1}\n"
        for key in keys:
            s += f"{key}: {logs[key]:.4f} - "
        print(s[:-3], file=open('log.txt', 'a'))

callback = tf.keras.callbacks.ModelCheckpoint(filepath="net/checkpoints/",
                                                verbose=1,
                                                save_freq='epoch')

TRAIN_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential([
        layers.InputLayer(input_shape = input_shape),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomContrast(0.5),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.3, width_factor=0.3),
        layers.experimental.preprocessing.RandomZoom(height_factor=0.3),
        layers.experimental.preprocessing.RandomCrop(128,128)
        ])
        #128x128x3
        self.model.add(layers.Conv2D(8, 13, activation = 'relu'))
        self.model.add(layers.BatchNormalization())
        #116x116x8
        self.model.add(layers.MaxPooling2D(pool_size = 2))
        #58x58x8
        self.model.add(layers.Conv2D(16, 11, activation = 'relu'))
        #48x48x16
        self.model.add(layers.MaxPooling2D(pool_size = 2))
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
        print(summaryStr, file=open('log.txt', 'w'))

if __name__ == "__main__":
    faceDirs = os.listdir('datasets/faces/')
    random.shuffle(faceDirs)
    faces = []
    for dir in faceDirs:
        faces += [cv2.imread('datasets/faces/'+dir+'/'+file) for file in os.listdir('datasets/faces/'+dir)]
        if len(faces) > 11000:
            break
    random.shuffle(faces)
    trainFX = faces[:8000]
    testFX = faces[8000:11000]
    hands = [cv2.imread('datasets/hands/'+file) for file in os.listdir('datasets/hands/')]
    random.shuffle(hands)
    trainHX = hands[:8000]
    testHX = hands[8000:11000]
    for i in range(len(trainFX)):
        trainFX[i] = [trainFX[i], np.array([1,0])]
    for i in range(len(testFX)):
        testFX[i] = [testFX[i], np.array([1,0])]
    for i in range(len(trainHX)):
        trainHX[i] = [trainHX[i], np.array([0,1])]
    for i in range(len(testHX)):
        testHX[i] = [testHX[i], np.array([0,1])]
    train = trainFX+trainHX
    random.shuffle(train)
    test = testFX+testHX
    random.shuffle(test)
    trainX = np.array([item[0] for item in train])
    trainY = np.array([item[1] for item in train])
    testX = np.array([item[0] for item in test])
    testY = np.array([item[1] for item in test])
    print(len(trainX), trainX[0].shape, trainY, len(testX), testX[0].shape, testY)
    load = input("Enter path to model to be loaded, or hit enter for no model: ")
    if load == "":
        net = Net((128, 128, 3)).model
    else:
        net = tf.keras.models.load_model(load)
    results = net.fit(x=generator(BATCH_SIZE_TRAIN, [trainX,trainY]),
                        validation_data=generator(BATCH_SIZE_TEST, [testX, testY]),
                        shuffle = True,
                        epochs = TRAIN_EPOCHS,
                        batch_size = BATCH_SIZE_TRAIN,
                        validation_batch_size = BATCH_SIZE_TEST,
                        verbose = 1,
                        steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN,
                        validation_steps=len(testX)/BATCH_SIZE_TEST,
                        callbacks=[callback, CustomCallback()]) #saving and logging
    tf.keras.models.save_model(net, "net/final/")
