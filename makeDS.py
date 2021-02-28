import cv2
import os
import random
import numpy as np

bgs = []

for file in [x for x in os.listdir('datasets/backgrounds/indoor/') if '.db' not in x]:
    print('datasets/backgrounds/indoor/'+file)
    img = cv2.imread('datasets/backgrounds/indoor/'+file)
    w = img.shape[1]
    h = img.shape[0]
    if w > h:
        img = img[:h, :h]
    elif h > w:
        img = img[:w, :w]
    img = cv2.resize(img, (128,128))
    cv2.imwrite('datasets/backgrounds/indoor/'+file, img)
    bgs.append('datasets/backgrounds/indoor/'+file)

for file in [x for x in os.listdir('datasets/backgrounds/outdoor/') if '.db' not in x]:
    img = cv2.imread('datasets/backgrounds/outdoor/'+file)
    w = img.shape[1]
    h = img.shape[0]
    if w > h:
        img = img[:h, :h]
    elif h > w:
        img = img[:w, :w]
    img = cv2.resize(img, (128,128))
    cv2.imwrite('datasets/backgrounds/outdoor/'+file, img)
    bgs.append('datasets/backgrounds/outdoor/'+file)

for file in os.listdir('datasets/hands/')[0:1]:
    img = cv2.imread('datasets/hands/'+file)
    cv2.imshow("i", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("i", img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    smth, thresh = cv2.threshold(img, 230, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    thresh = cv2.GaussianBlur(thresh, (5,5), 0)
    smth, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("t1", thresh)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    bg = cv2.imread(random.choice(bgs))
    cv2.imshow("bg", bg)
    thresh = cv2.add(thresh, bg)
    cv2.imshow("t2", thresh)
    thresh[np.where(thresh == 255)] = 0
    cv2.imshow("t3", thresh)
    img = cv2.imread('datasets/hands/'+file)
    img[np.where(thresh != 0)] = 0
    img = cv2.add(img, thresh)
    cv2.imshow("final", img)
    cv2.imwrite('datasets/hands/'+file, img)
