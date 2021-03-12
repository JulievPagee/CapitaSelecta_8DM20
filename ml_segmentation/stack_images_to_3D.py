# -*- coding: utf-8 -*-
"""
Created on 12/03/2021

@author: 20165272
"""
import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pylab as plt
from glob import glob
import argparse
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import mahotas as mt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
import pandas as pd

#option 1
#ref: https://stackoverflow.com/questions/31573872/how-to-read-multiple-images-and-create-a-3d-matrix-with-them
#arrays = []
#for number in range(0, 299):
#    numstr = str(number).zfill(3)
 #   fname = numstr + '.bmp'
 #   a = imread(fname, flatten=1)
 #   arrays.append(a)
#data = np.array(arrays)
#dit lijkt me vrij omslachtig, np.concatenate doet ook de job.

#option 2
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.dstack.html
#a = figure1   #np.array format
#b = figure2   #np.array format
#np.concatenate((a, b), axis=0)  #stackt ze in verticale richting (links =1, rechts =2)
#np.concatenate((a, b.), axis=1) #stackt ze in horizontale richting
#np.concatenate((a,b) axis=2)    #axis 2 obv bovenste redenatie
#option 3
#https://nilearn.github.io/modules/generated/nilearn.image.concat_imgs.html



#doel (333,271,86) array maken.
#option 2 lijkt me het handiste
def slice_to_3D(data_list, save_path):

    final_image = np.array([])
    final_image = np.zeros((333,271,86))
    firstIteration = True
    for item in data_list:
        img = data_path+ patient_path + item + '.png'
        img = sitk.ReadImage(img)
        img = sitk.GetArrayFromImage(img)
        if firstIteration:
            final_image = img
            firstIteration=False
        else:
            final_image = np.concatenate(([img, final_image ]), axis=2)
            #error: axis 2 is out of bounds for array of dimension 2
            # wanneer ik axis = 1 gebruik komen ze naast elkaar te staan.

    final_image= Image.fromarray(final_image)
    final_image.show()

data_path = 'C:/Users/20165272/Documents/8DM20 Capita Selecta/Project/ml_segmentation_old/ClassData/Labelled/Slices/'
patient_path = 'img_p107_slice_'
data_list = [ '1', '2', '3']
save_path= 'C:/Users/20165272/Documents/8DM20 Capita Selecta/Project/ml_segmentation_old'
save_name= 'final_image'

slice_to_3D(data_list,save_path)