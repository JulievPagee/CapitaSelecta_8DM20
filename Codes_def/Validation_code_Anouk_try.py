# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:39:20 2021

@author: 20161879
"""
from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import torch
from Validation_utilities import *

######################################################################################################
# set Codes_def path
codes_def_path = r'C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\TrainingData'

#elastix and transformix paths   
ELASTIX_PATH = r"C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\Elastix\elastix.exe"
TRANSFORMIX_PATH = r"C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\Elastix\elastix.exe"
######################################################################################################
# create resutls folder
results_dir = codes_def_path+'Results_atl_reg/'

if os.path.exists(results_dir) is False:
            os.mkdir(results_dir)
            
path_val = codes_def_path+'NormData/Validation/'
path_atlas = codes_def_path+'NormData/Atlas/'
parameter_path = codes_def_path+'Parameter_files/'
DATA_DIR = codes_def_path+'TrainingData/'

###create val and atlas list###
val_data = []
Val_labels = ['p133', 'p119', 'p125']
for dirName, subdirList, fileList in os.walk(path_val):
    for filename in fileList:
        if '.mhd' in filename:
            val_data.append(os.path.join(dirName,filename))
            
atlas_data = []
Atlas_labels = ['p107', 'p108', 'p116', 'p127', 'p135']
for dirName, subdirList, fileList in os.walk(path_atlas):
    for filename in fileList:
        if '.mhd' in filename:
            atlas_data.append(os.path.join(dirName,filename))        
            
###empty lists for saving the new masks and the paths###
segm_images = []
segm_images_paths = []
img_atlas = []

###transforms the atlas masks based on best parameter file###
#loop over val data (n=3)
for val_idx in range(len(val_data)):
    #loop over atlasses (n=5)
    for atlas_idx in range(len(atlas_data)):
        #set and create neededpaths
        fixed_image_path = val_data[val_idx]
        moving_image_path = atlas_data[atlas_idx]
        
        pnr = Val_labels[val_idx]
        anr = Atlas_labels[atlas_idx]
        
        registered_image_path, image_registered = registration_func(pnr, anr, results_dir, ELASTIX_PATH, 
                                                                    fixed_image_path, moving_image_path, parameter_path)
    
    #get the transformed atlasses
    t_atlasses_dir = results_dir+'results_B_spline_{}'.format(pnr)
    t_atlasses = os.listdir(t_atlasses_dir)
    #get segmentation of the best resembling atlas based on mutual information
    atlas_seg, atlas_seg_path, idx = GetBestAtlasSegmentationFile(DATA_DIR, t_atlasses_dir, fixed_image_path, t_atlasses, Atlas_labels)
    
    
    #path where transformed image is saved
    dir_res = results_dir + 'transform_{}/'.format(pnr)
    if os.path.exists(dir_res) is False:
            os.mkdir(dir_res)
    #apply the transformation to the mask of the best atlas
    atlas_seg_path, transformed_atlas_path = deform_func(results_dir, pnr, segm_img, t_atlasses[idx], dir_res, TRANSFORMIX_PATH)
    
    #store the transformed mask and path for further use
    segm_images.append(segm_img)
    segm_images_paths.append(transformed_atlas_path)

    img_atlas.append([Val_labels[val_idx], Atlas_labels[idx]])

### lists of list where the first element is the val img and the second element the best atlas###
print(img_atlas)

###visualize result###
result_folders = os.listdir(results_dir)
#index in img_atlas [0-2]
img = 0

#create images
atlas_seg_path = DATA_DIR+img_atlas[img][1]+'/prostaat.mhd'
atlas_seg = imageio.imread(atlas_seg_path)
atlas_seg= torch.from_numpy(atlas_seg).permute(1,2,0)

pred_seg_path = 'E:/CSMIA/Codes_def/Results_atl_reg/transform_p119/Transformed_masks/mask_B_spline_p119/result.mhd'
pred_seg = imageio.imread(pred_seg_path)
pred_seg = torch.from_numpy(pred_seg).permute(1,2,0)

gt_seg_path = DATA_DIR+img_atlas[img][0]+'/prostaat.mhd'
gt_seg = imageio.imread(gt_seg_path)
gt_seg = torch.from_numpy(gt_seg).permute(1,2,0)

# Plot with scroll function (does not work for some reason)
plot_segm(atlas_seg, pred_seg, gt_seg)

# Evaluation
#gt_seg and pred_seg are renamed to get a clear interpretation about test en predicted
y_test = gt_seg
y_pred = pred_seg

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(y_test, y_pred)
dice = dice(y_pred, y_test)
print ('Sensitivity:', sensitivity)
print ('Specificity:', specificity)
print ('Accuracy:', accuracy)
print('Dice:', dice)

