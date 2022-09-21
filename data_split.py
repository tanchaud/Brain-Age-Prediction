#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:30:14 2018

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
 
path = "/home/chaudhuri/Data/brainage_new/IXI/"

Images,Labels,Rid = data_convert(path)
#D = data_split(Images,Labels,Rid)


