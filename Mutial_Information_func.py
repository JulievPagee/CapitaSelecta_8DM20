# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:03:18 2021

@author: 20161879
"""

from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import imageio

#Functie die de path van de beste segmentatie teruggeeft.
def GetBestAtlasSegmentationFile(fixed_img_path, atlasses):
    #Functie om Mutual Information te bepalen
    def mutual_information(fixed_img, atlas_img):
        """ Mutual information for joint histogram
        """
        hist_2d, x_edges, y_edges = np.histogram2d(fixed_img.ravel(), atlas_img.ravel(), bins=20)
        # Convert bins counts to probability values
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    #Mutual information berekenen
    MIlist = []
    for i in range(5):
        atlas_path = os.path.join(DATA_DIR, atlasses[i], 'mr_bffe.mhd')
        fixed_img = imageio.imread(fixed_img_path)
        atlas_img = imageio.imread(atlas_path)
        
        MI = mutual_information(fixed_img, atlas_img)
        MIlist.append(MI)
    # Index vinden van de atlas met de hoogste mutual information
    index = MIlist.index(max(MIlist))
    
    #De binary file van de beste atlas inladen
    atlas_seg_path = os.path.join(DATA_DIR, atlasses[index], 'prostaat.mhd')
    atlas_seg = imageio.imread(atlas_seg_path)
    
    return atlas_seg, atlas_seg_path
    
#Data inladen
DATA_DIR = 'E:/CSMIA/TrainingData/'
patient_fold = os.listdir(DATA_DIR)

#Split moet nog randomized
atlasses = patient_fold[10:15]
training = patient_fold[:10]

patient_nr = 1

fixed_img_path = os.path.join(DATA_DIR, training[patient_nr], 'mr_bffe.mhd')
fixed_seg_path = os.path.join(DATA_DIR, training[patient_nr], 'prostaat.mhd')

atlas_seg, atlas_seg_path = GetBestAtlasSegmentationFile(fixed_img_path, atlasses)

#Dice score berekenen
gt = imageio.imread(fixed_seg_path)
seg = atlas_seg
dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))

