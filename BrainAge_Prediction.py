#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:34:58 2017

@author: chaudhuri
"""

from data import data_convert,data_split
import nibabel as nib
import glob
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from sklearn import svm

#%% Data

path = "/home/chaudhuri/Data/brainage_new/IXI/"
Images,Labels,Rid = data_convert(path)
#D = data_split(Images,Labels,Rid)
X_train, X_test, y_train, y_test = train_test_split(Images, Labels, test_size=0.25)

# Data Visualisations
unique_data,counts_data = np.unique(Labels,return_counts=True)
plt.bar(unique_data,counts_data, label='IXI data')
plt.xlabel('Age')
plt.ylabel('Counts')
unique_train,counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train,counts_train, label='train')
unique_test,counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_test,counts_test, label='test')
#plt.legend()
plt.show()

#%% Feature Extraction

imgarray = np.empty([145,121],np.float32)
ImageArray = Images

for j in val,enumerate(ImageArray):
    ImageObj = ImageArray[j]

[x,y,z] = ImageObj.shape
slices = x

total_modes = 121/3
i = 1
FM = np.array([],dtype=uint8)
    for k in val,enumerate(img[j]):
        mode[k] = cat(img(j)[:,:,i], img(j)[:,:,i+1], img(j)[:,:,i+2])
        # preprocessing steps
        mode[i] = preprocess(mode[i])
        FV[i] = VGG16_model(mode[i])        # shape(FV[i]) = [1,9216] for eg.  
        FM = FM.append(FV[i])
        i = i+3


        