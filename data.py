#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:06:09 2018

@author: chaudhuri
"""

import nibabel as nib
import glob
import numpy as np
import scipy.io as sio

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
#%%    
    
def data_split(Images,Labels,Rid):
    
    sorted_age = np.sort(Labels,axis = 0)
    idx = [i[0] for i in sorted(enumerate(Labels), key=lambda x:x[1])]
    sorted_subjects = Rid[idx]
    
    TRAIN1_3_Rid = np.array([], dtype = np.uint8)
    TRAIN1_3_Labels = np.array([], dtype = np.uint8)
    TEST1_3_Rid = np.array([], dtype = np.uint8)
    TEST1_3_Labels = np.array([], dtype = np.uint8)
    
    # Image IDs
    for ix, val in enumerate(sorted_subjects):
        
        if ix % 4 == 0:
            TEST1_3_Rid = np.append(TEST1_3_Rid, sorted_subjects[ix])
            TEST1_3_Labels = np.append(TEST1_3_Labels, sorted_age[ix])
            
        else:
            TRAIN1_3_Rid = np.append(TRAIN1_3_Rid, sorted_subjects[ix])
            TRAIN1_3_Labels = np.append(TRAIN1_3_Labels, sorted_age[ix])
            
        D = TRAIN1_3_Rid,TRAIN1_3_Labels,TEST1_3_Rid,TEST1_3_Labels    
        return D    
