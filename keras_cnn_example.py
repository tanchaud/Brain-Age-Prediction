#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:16:17 2018

@author: chaudhuri
"""
#%%  Libraries and Modules

import numpy as np 
np.random.seed(123) # reproducibility

# importing sequential model from Keras: a linear stack of neural network layers
# for feed-forward CNN that is being built in this script
from keras.models import Sequential 

#import core layers. i.e. layers used in almost any neural network
from keras.layers import Dense, Dropout, Activation, Flatten

# import CNN layers: convolutional layers for training on image data
from keras.layers import Convolution2D, MaxPooling2D

# utilities, to help transform data later
from keras.utils import np_utils

#%% load data

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape

#%% preprocessing steps 

# -----------preprocess input images--------- # 

# reshape input data: to declare depth 
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print X_train.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# -----------preprocess class labels--------- # 
print y_train.shape
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#%% model architecture

# declare sequential model 
model = Sequential()

# CNN input layer
model.add(Convolution2D(32, 3, 3, activation = 'relu', input_shape=(1,28,28), dim_ordering = 'th'))
print model.output_shape

# add more layers
model.add(Convolution2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# To complete architecture: fully connected dense layers 
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

#%% compile model: define loss function and optimizer

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#%% fit model on training data: declare batch size and number of epochs, pass training data

model.fit(X_train, Y_train, 
          batch_size = 32, nb_epoch = 10, verbose = 1)

#%% evaluate model on test set

score = model.evaluate(X_test, Y_test, verbose = 0)




