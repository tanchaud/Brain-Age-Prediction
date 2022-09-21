#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:43:39 2017

@author: chaudhuri
"""

#import os
import nibabel as nib
import glob
import numpy as np
import scipy.io as sio

#%%
def data_convert(path):
        
#    path = "/home/chaudhuri/Data/brainage_new/IXI/"
        
    # 1. paths to images and labels
    path_images = path + "*.nii"
    path_labels = path + "tables/AGE_IXI547.mat"
    path_rid = path + "tables/RID_IXI547.mat"
#    path_labels = path + "labels/c108_age.mat"
    
    # 2. load labels & rid
    age = sio.loadmat(path_labels)
    Y = age['age_IXI547']
    Y = Y[:100]
    rid = sio.loadmat(path_rid)
    rid = rid['rid']
    rid = rid[:100]
    
    # 3. load images
    images_Abspath = glob.glob(path_images)[:100]
    imagearray = np.zeros((len(Y), 121, 145, 121), dtype=np.float32)
    for i, imagepath in enumerate(images_Abspath):
        img = nib.load(imagepath)
        imagearray[i] = img.get_data()
        
        Data = imagearray,Y,rid
                
        return Data
        
# FUNCTION CALL 
path = "/home/chaudhuri/Data/brainage_new/IXI/"
Images,Labels,Rid = data_convert(path)