#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:45:04 2018

@author: chaudhuri
"""
#%%---------To address overfitting-------#

#1. optimize hyperparameters 
# This is optional, as the "best" C has been found on matlab and has not helped in prediction.
# however, could be re-implemented using built-in functions on python to be absolutely sure.

#2. Regularization of linear SVM 


#%%
from pytictoc import TicToc
import os
import scipy.io as sio
#import glob
#import numpy as np
#from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
#import pandas as pd
#import matplotlib.pyplot as plt

#%%

# load linear SVR model with default hyperparameter settings
path_models = "/home/chaudhuri/code/python/models/"
os.chdir(path_models)


