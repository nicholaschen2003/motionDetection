import numpy as np
import random
import tensorflow.keras.models as models
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import cv2

netF = models.load_model('net/final/')
netC = models.load_model('net/checkpoints/')

img = np.random.rand(1, 128, 128, 3)

print("F", netF.predict(img))
print("C", netC.predict(img))
cv2.imshow("Image", img.reshape(128,128,3))
cv2.waitKey(0)
