#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:37:13 2018

@author: chaudhuri
"""

#-----Extract features with VGG16------#

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from data import data_convert

#
#model = VGG16(weights='imagenet', include_top=False)
#
#img_path = "/home/chaudhuri/Data/paintings/original/Aalto1.jpg"
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

#features = model.predict(x)

path = "/home/chaudhuri/Data/brainage_new/IXI"
Images,Labels,Rid = data_convert(path)

