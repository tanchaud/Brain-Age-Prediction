#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:31:35 2018

@author: chaudhuri
"""

## MODES VISUALIZATION FOR FUSION SCHEMES.....

#Packages 
#from data import data_convert
from data import data_convert
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import cv2

#Images 
#path = os.getcwd()
##Images,Labels,Rid = data_convert(path)
#Images = nib.load(path + '/rp1IXI002-Guys-0828-T1_affine.nii')
path = "/home/chaudhuri/Data/brainage_new/IXI/"
Images,Labels,Rid = data_convert(path)
Images = Images.get_data() # convert object iimage to array image
print Images.shape

mriImage = Images[1]
print mriImage.shape

# Modes
total_modes = 40
mid_mode = np.stack((mriImage[:,:,59],mriImage[:,:,60],mriImage[:,:,61]), axis=2)
plt.imshow(mid_mode)
plt.imshow(mriImage[:,:,60])
print sum(sum(mriImage[:,:,60]))

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
    slice_0 = Images[26, :, :]
    slice_1 = Images[:, 30, :]
    slice_2 = Images[:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for EPI image")  


slice_1_r = Images[:, 59, :]
slice_1_g = Images[:, 60, :]
slice_1_b = Images[:, 61, :]
img = cv2.merge((slice_1_r,slice_1_g,slice_1_b))



