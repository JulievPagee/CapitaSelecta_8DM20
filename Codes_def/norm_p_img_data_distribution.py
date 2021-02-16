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
DATA_DIR = 'TrainingData' # adjust to your path of TrainingData
SAVE_FOLD = 'NormData/'
SAVE_FOLD_V = 'NormData/Validation/'
SAVE_FOLD_T = 'NormData/Train/'
SAVE_FOLD_A = 'NormData/Atlas/'

# data distribution randomly chosen
Val = ['p133', 'p119', 'p125']
Train = ['p102', 'p109', 'p115', 'p117', 'p120', 'p128', 'p129']
Atl = ['p107', 'p116', 'p135', 'p127', 'p108']


if os.path.exists(SAVE_FOLD) is False:
    os.mkdir(SAVE_FOLD)

if os.path.exists(SAVE_FOLD_V) is False:
    os.mkdir(SAVE_FOLD_V)
if os.path.exists(SAVE_FOLD_T) is False:
    os.mkdir(SAVE_FOLD_T)
if os.path.exists(SAVE_FOLD_A) is False:
    os.mkdir(SAVE_FOLD_A)

patient_fold = os.listdir(DATA_DIR)

for j in range(15):
    # select the data category of each image
    if patient_fold[j] in Val:
        category = 'Validation'
    elif patient_fold[j] in Train:
        category = 'Train'
    elif patient_fold[j] in Atl:
        category = 'Atlas'
    SAVE_FOLD = 'NormData/{}/'.format(category)
    
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

print("Images are normalized")