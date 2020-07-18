# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:54:48 2018

@author: NaSiF
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
from scipy import misc
import cv2

digits = datasets.load_digits()

#print(type(digits.data[0]))
#print(digits.data[0])
#print(digits.images[-10])
#print("\nNumber Of images Used=> ",len(digits.data))

clf = SVC(gamma = 0.001, C=100)
X,Y= digits.data[:-10], digits.target[:-10]
clf.fit(X,Y)

print('\nPrediction:',clf.predict(digits.data[-4].reshape(1,-1)))


plt.imshow(misc.imresize(digits.images[-4],(128,128)), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()