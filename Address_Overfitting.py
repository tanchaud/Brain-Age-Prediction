#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:45:04 2018

@author: chaudhuri
"""
#%%---------To address overfitting-------#

#1. Hyperparameter Optimization using grid search. (KFCV.m) 
#  http://scikit-learn.org/stable/modules/grid_search.html (python code tutorial)

#2. Regularization of linear SVM : https://www.kaggle.com/apapiu/regularized-linear-models


#%%
from pytictoc import TicToc
import os
import scipy.io as sio
#import glob
import numpy as np
#from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

#%% --------- Features and Labels----------%

path_train = "/home/chaudhuri/code/matlab/CNN_Features_BrainAge_10-4-2018/Training/"
os.chdir(path_train)
X_train = sio.loadmat('VGG-VD-16_IXI(Training)_fused.mat', squeeze_me=True)
X_train = X_train['FM']
Y_train = sio.loadmat('Training_labels.mat', squeeze_me=True)
Y_train = Y_train['TS_labels']

#%% 1. Hyperparameter Optimization of linear SVR model ----% 

## load linear SVR model with default hyperparameter settings
#path_models = "/home/chaudhuri/code/python/models/"
#os.chdir(path_models)
#svr_lin = joblib.load(os.path.join(path_models,'svr_lin.sav'))
#model_params = svr_lin.get_params()
#
## Set the parameters by cross-validation
#tuned_params = [{'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]}]
#
#scores = []

#%% 2. Other linear regression models------% 

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
path_models = "/home/chaudhuri/code/python/models/"
t = TicToc()
t.tic() #Start Timer

#model_ridge = Ridge()
#alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
#            for alpha in alphas]
#cv_ridge = pd.Series(cv_ridge, index = alphas)
#cv_ridge_filename = os.path.join(path_models,'cv_ridge.sav')
#cv_ridge = joblib.dump(cv_ridge,cv_ridge_filename)
#cv_ridge = joblib.load(cv_ridge_filename)
#cv_ridge.plot(title = "Validation - Just Do It")
#plt.xlabel("alpha")
#plt.ylabel("rmse")

model_lasso = LassoCV(alphas = [0.05, 0.1, 0.3, 1]).fit(X_train, Y_train)
cv_lasso = rmse_cv(model_lasso).mean()
print cv_lasso

t.toc() #Time elapsed since t.tic()

