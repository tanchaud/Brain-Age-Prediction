#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:47:18 2018

@author: chaudhuri
"""
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

#%%-----features and labels-----#

# 1. load training set features & labels
path_train = "/home/chaudhuri/code/matlab/CNN_Features_BrainAge_10-4-2018/Training/"
os.chdir(path_train)
X_train = sio.loadmat('VGG-VD-16_IXI(Training)_fused.mat', squeeze_me=True)
X_train = X_train['FM']
Y_train = sio.loadmat('Training_labels.mat', squeeze_me=True)
Y_train = Y_train['TS_labels']

#%% ----train SVR model with default hyperparameters-----#

#path_models = "/home/chaudhuri/code/python/models/"
#os.chdir(path_models)
#
#t = TicToc()
#t.tic() #Start Timer
#svr_rbf = SVR(kernel='rbf', C=1.0, gamma='auto')
#Ypred_rbf = svr_rbf.fit(X_train, Y_train).predict(X_train)
#svr_rbf_filename = os.path.join(path_models,'svr_rbf.sav')
#joblib.dump(svr_rbf,svr_rbf_filename)#save trained model
#Ypred_rbf_filename = os.path.join(path_models,'Ypred_rbf.pkl')
#joblib.dump(Ypred_rbf,Ypred_rbf_filename)
#
#svr_lin = SVR(kernel='linear', C=1.0)
#Ypred_lin = svr_lin.fit(X_train, Y_train).predict(X_train)
#svr_lin_filename = os.path.join(path_models,'svr_lin.sav')
#joblib.dump(svr_lin,svr_lin_filename)
#Ypred_lin_filename = os.path.join(path_models,'Ypred_lin.pkl')
#joblib.dump(Ypred_lin,Ypred_lin_filename)
#
#svr_poly = SVR(kernel='poly', C=1.0, degree=3)
#Ypred_poly = svr_poly.fit(X_train, Y_train).predict(X_train)
#svr_poly_filename = os.path.join(path_models,'svr_poly.sav')
#joblib.dump(svr_poly,svr_poly_filename)
#Ypred_poly_filename = os.path.join(path_models,'Ypred_poly.pkl')
#joblib.dump(Ypred_poly,Ypred_poly_filename)
#t.toc() #Time elapsed since t.tic()

#%% ----predictions----#

#1. load validation/test sets features & labels
path_val = "/home/chaudhuri/code/matlab/CNN_Features_BrainAge_10-4-2018/Validation/"
os.chdir(path_val);
X_val = sio.loadmat('VGG-VD-16_IXI(Validation)_fused.mat', squeeze_me=True)
X_val = X_val['FM']
Y_val = sio.loadmat('Validation_labels.mat', squeeze_me=True)
Y_val = Y_val['Labels']

#2. load trained models & predict on validation
path_models = "/home/chaudhuri/code/python/models/"
#svr_rbf_filename = os.path.join(path_models,'svr_rbf.sav')
#svr_rbf = joblib.load(svr_rbf_filename)
svr_lin_filename = os.path.join(path_models,'svr_lin.sav')
svr_lin = joblib.load(svr_lin_filename)
#svr_poly_filename = os.path.join(path_models,'svr_poly.sav')
#svr_poly = joblib.load(svr_poly_filename)

#Ypred_val = svr_rbf.predict(X_val)
#print "Using rbf SVR, model performance on validation is..."
#print mean_absolute_error(Y_val,Ypred_val)

Ypred_train =svr_lin.predict(X_train)
print "Using linear SVR, mae on training is..."
print mean_absolute_error(Y_train,Ypred_train)
Ypred_val = svr_lin.predict(X_val)
print "Using linear SVR, mae on validation is..."
print mean_absolute_error(Y_val,Ypred_val)

#Ypred_val = svr_poly.predict(X_val)
#print "Using poly SVR, model performance on validation is..."
#print mean_absolute_error(Y_val,Ypred_val)

# Choose lowest mae and evaluate model's tendency to overfit.
# Lowest mae given by linear SVR model
# Linear SVR model is overfitting. 
    # 1.Find best C penalty using 5-fold cross validation -- done! 
    # 2.Regularization   

#%% Rough
#Ypred_val_filename = os.path.join(path_models,'Ypred_rbf_val.pkl')
#joblib.dump(Ypred_val,Ypred_val_filename)






