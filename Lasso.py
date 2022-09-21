#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:47:18 2018

@author: chaudhuri
"""
import os
import scipy.io as sio
import glob
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
#import pandas as pd
import matplotlib.pyplot as plt

#-----load features and labels as .mat files-----#

# 1. path to features and labels
path_train = "/home/chaudhuri/code/matlab/CNN_Features_BrainAge_10-4-2018/Training/"
os.chdir(path_train);
X_train = sio.loadmat('VGG-VD-16_IXI(Training)_fused.mat', squeeze_me=True)
X_train = X_train['FM']
Y_train = sio.loadmat('Training_labels.mat', squeeze_me=True)
Y_train = Y_train['TS_labels']

#path_val = "/home/chaudhuri/code/matlab/CNN_Features_BrainAge_10-4-2018/Validation/"


# train SVR model for age prediction

