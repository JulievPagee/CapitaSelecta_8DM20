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
from deform_func import *

######################################################################################################
# set Codes_def path
codes_def_path = 'E:/CSMIA/Codes_def/'
# create resutls folder
results_dir = 'E:/CSMIA/Results_atl_reg/'
if os.path.exists(results_dir) is False:
    os.mkdir(results_dir)
#elastix and transformix paths   
ELASTIX_PATH = "E:/CSMIA/elastix.exe"
TRANSFORMIX_PATH = "E:/CSMIA/transformix.exe"
######################################################################################################

path_val = codes_def_path+'NormData/Validation/'
path_atlas = codes_def_path+'NormData/Atlas/'
parameter_path = codes_def_path+'Parameter_files/parameters.txt'
DATA_DIR = codes_def_path+'TrainingData/'

#create val and atlas list
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
            
#empty lists for saving the new masks and the paths
segm_images = []
segm_images_paths = []
img_atlas = []

#transforms the atlas masks based on best parameter file
#loop over val data (n=3)
for val_idx in range(len(val_data)):
    #loop over atlasses (n=5)
    for atlas_idx in range(len(atlas_data)):
        #set and create neededpaths
        fixed_image_path = val_data[val_idx]
        moving_image_path = atlas_data[atlas_idx]
        t_val_dir = results_dir+'Atlas_reg_{}/'.format(Val_labels[val_idx])
        output_dir = t_val_dir+'atlas_{}'.format(Atlas_labels[atlas_idx])
        
        if os.path.exists(t_val_dir) is False:
            os.mkdir(t_val_dir)
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        #perform registration and save the images
        el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
        el.register(
        fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=output_dir)
    #get the transformed atlasses
    t_atlasses = os.listdir(t_val_dir)
    #get segmentation of the best resembling atlas based on mutual information
    atlas_seg, atlas_seg_path, idx = GetBestAtlasSegmentationFile(DATA_DIR, t_val_dir, fixed_image_path, t_atlasses, Atlas_labels)
    
    #get transform parameter file applied to the best atlas     
    par_path = os.path.join(t_val_dir,os.listdir(t_val_dir)[idx], 'TransformParameters.0.txt')
    
    #path where transformed image is saved
    dir_res = t_val_dir + 'transform'
    if os.path.exists(dir_res) is False:
            os.mkdir(dir_res)
    #apply the transformation to the mask of the best atlas
    segm_img, transformed_atlas_path = apply_transform(TRANSFORMIX_PATH, dir_res, atlas_seg_path, par_path)
    
    #store the transformed mask and path for further use
    segm_images.append(segm_img)
    segm_images_paths.append(transformed_atlas_path)

    img_atlas.append([Val_labels[val_idx], os.listdir(t_val_dir)[idx][6:]])

# lists of list where the first element is the val img and the second element the best atlas
print(img_atlas)

#visualize result
result_folders = os.listdir(results_dir)
#index in img_atlas [0-2]
img = 0

#create images
atlas_seg_path = DATA_DIR+img_atlas[img][1]+'/prostaat.mhd'
atlas_seg = imageio.imread(atlas_seg_path)
atlas_seg= torch.from_numpy(atlas_seg).permute(1,2,0)

pred_seg_path = results_dir+result_folders[img]+'/transform/result.mhd'
pred_seg = imageio.imread(pred_seg_path)
pred_seg = torch.from_numpy(pred_seg).permute(1,2,0)

gt_seg_path = DATA_DIR+img_atlas[img][0]+'/prostaat.mhd'
gt_seg = imageio.imread(gt_seg_path)
gt_seg = torch.from_numpy(gt_seg).permute(1,2,0)

# Plot with scroll function (does not work for some reason)
plot_segm(atlas_seg, pred_seg, gt_seg)
