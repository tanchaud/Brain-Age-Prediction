#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:31:56 2017

@author: chaudhuri
"""


import os
import numpy as np

from nibabel.testing import data_path

example_filename = os.path.join(data_path, 'example4d.nii.gz')
import nibabel as nib 
img = nib.load(example_filename)

