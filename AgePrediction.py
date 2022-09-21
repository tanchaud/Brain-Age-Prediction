#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:16:00 2017

@author: chaudhuri
"""

import os
#import sys
import glob #Filename pattern matching library
import nibabel as nib

def data_convert(path):

#Adding paths of images and labels to Python's current directory
    #sys.path.append("/home/chaudhuri/Data/brainage_new/IXI") 
    #sys.path.append('~/Data/brainage_new/Data_for_model/')
#Creating directory of nifti files 
#path = "/home/chaudhuri/Data/brainage_new/IXI/"
    os.chdir(path)
    imageList = glob.glob('*.nii')       
    mri = []
#Loading the nifti images 
    for x in imageList:      
        img = nib.load(x)
        #mri.append = img.get_data()
        mri.append(img.get_data())
    return mri        

path = "/home/chaudhuri/Data/brainage_new/IXI/"

mri = data_convert(path)




    