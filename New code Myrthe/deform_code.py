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

dir_res = 'E:/CSMIA/resultstest5/'
moving_image_path = 'E:/CSMIA/NormData/norm_img_1.mhd'
atlas_segm_path = 'E:/CSMIA/TrainingData/p109/prostaat.mhd'
ground_truth_path = 'E:/CSMIA/TrainingData/p108/prostaat.mhd'

def apply_transform(dir_res, moving_image_path, atlas_segm_path):
    TRANSFORMIX_PATH = os.path.join(r'transformix.exe')
    # Make a new transformix object tr with the CORRECT PATH to transformix
    tr = elastix.TransformixInterface(parameters=os.path.join(dir_res, 'TransformParameters.0.txt'),
                                      transformix_path=TRANSFORMIX_PATH)
    
    # Transform a new image with the transformation parameters
    transformed_image_path = tr.transform_image(moving_image_path, output_dir=dir_res)
    
    # Get the Jacobian matrix
    jacobian_matrix_path = tr.jacobian_matrix(output_dir=dir_res)
    
    # Get the Jacobian determinant
    jacobian_determinant_path = tr.jacobian_determinant(output_dir=dir_res)
    
    # Get the full deformation field
    deformation_field_path = tr.deformation_field(output_dir=dir_res)
    
    deform_field = imageio.imread(deformation_field_path)
    
    segm_at = imageio.imread(atlas_segm_path)
    #create 4th axis
    segm_at = segm_at[:,:,:,np.newaxis]
    segm_at = np.concatenate((segm_at, np.ones((86,333,271,2))), axis=-1)
    #apply deformation
    segm_img = segm_at*deform_field
    #remove 4th axis
    segm_img = segm_img[:,:,:,0]
    
    return segm_img

segm_img = apply_transform(dir_res, moving_image_path, atlas_segm_path)
segm_img[segm_img != 0] = 1

orig = imageio.imread(atlas_segm_path)
gt = imageio.imread(ground_truth_path)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)

ax0.imshow(orig[20], cmap='gray')
ax0.set_title('Original (atlas) mask')
ax1.imshow(segm_img[20], cmap='gray')
ax1.set_title('Transformed (atlas) mask')
ax2.imshow(gt[20], cmap='gray')
ax2.set_title('Ground Truth')


