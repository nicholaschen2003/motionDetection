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

faceDirs = os.listdir('datasets/faces/')
random.shuffle(faceDirs)
faces = []
for dir in faceDirs:
    faces += [cv2.imread('datasets/faces/'+dir+'/'+file) for file in os.listdir('datasets/faces/'+dir)]
    if len(faces) > 11000:
        break
random.shuffle(faces)
hands = [cv2.imread('datasets/hands/'+file) for file in os.listdir('datasets/hands/')]
random.shuffle(hands)
test = faces+hands
random.shuffle(test)
net = tf.keras.models.load_model('net/checkpoints/')
for pic in test:
    result = net.predict(pic.reshape(1,128,128,3))
    if result[0][0] > result[0][1]:
        print("face")
    else:
        print("hand")
    cv2.imshow("test", pic)
    cv2.waitKey(0)

# face = cv2.resize(cv2.imread('datasets/face.png'), (128,128)).reshape(1,128,128,3)
# hand = cv2.resize(cv2.imread('datasets/hand.png'), (128,128)).reshape(1,128,128,3)
# net = tf.keras.models.load_model('net/checkpoints/')
# result = net.predict(face)
# if result[0][0] > result[0][1]:
#     print("face")
# else:
#     print("hand")
# cv2.imshow("test", face[0])
# cv2.waitKey(0)
# result = net.predict(hand)
# if result[0][0] > result[0][1]:
#     print("face")
# else:
#     print("hand")
# cv2.imshow("test", hand[0])
# cv2.waitKey(0)
