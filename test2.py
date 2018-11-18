# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:22:57 2018

@author: Wen Jie
"""
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# RUN THIS AFTER RUNNING test.py
# Code to plot a shape of chess moves sequence
# Input index manually
# If an error is thrown, means index not inside training set. Try a new index

X_train2 = X_train * 255

X_train2 = X_train2.reshape(34997, 18,18)

index = 7822
fig = plt.figure()
plt.subplot(2,1,1)
plt.imshow(X_train2[index], cmap='gray', interpolation='none')
plt.title("Result: {}".format(y_train2[index]))