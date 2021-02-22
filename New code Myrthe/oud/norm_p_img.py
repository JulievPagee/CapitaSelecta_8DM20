# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:14:47 2021

@author: 20161879
"""
import numpy as np
import imageio
import os
import SimpleITK as sitk

#load data
DATA_DIR = 'C:/Users/20175722/Documents/Master jaar 1/Kwartiel 3/8DM20 - CS/Group project/TrainingData'
SAVE_FOLD = 'NormData/'

if os.path.exists(SAVE_FOLD) is False:
    os.mkdir(SAVE_FOLD)

patient_fold = os.listdir(DATA_DIR)

for j in range(15):
    #load image
    img_path = os.path.join(DATA_DIR, patient_fold[j], 'mr_bffe.mhd')
    img = imageio.imread(img_path)
    #standardization step
    mean = img.mean(axis=(0,1,2))
    std = img.std(axis=(0,1,2))
    image = (img-mean)/std
    #save images
    save_path = SAVE_FOLD+'norm_img_'+patient_fold[j]+'.mhd'
    image = sitk.GetImageFromArray(image)
    image = sitk.WriteImage(image,save_path)

print("Images normalized")