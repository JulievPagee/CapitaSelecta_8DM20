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
import SimpleITK as sitk

dir_res = 'E:/CSMIA/resultstest12/'
dir_res_t = 'E:/CSMIA/resultstest12/transf/'
atlas_segm_path = 'E:/CSMIA/TrainingData/p109/prostaat.mhd'
ground_truth_path = 'E:/CSMIA/TrainingData/p108/prostaat.mhd'

if os.path.exists(dir_res_t) is False:
    os.mkdir(dir_res_t)
    
def apply_transform(dir_res, atlas_segm_path):
    TRANSFORMIX_PATH = os.path.join(r'transformix.exe')
    # Make a new transformix object tr with the CORRECT PATH to transformix
    tr = elastix.TransformixInterface(parameters=os.path.join(dir_res, 'TransformParameters.0.txt'),
                                      transformix_path=TRANSFORMIX_PATH)
    
    # Transform a new image with the transformation parameters
    transformed_atlas_path = tr.transform_image(atlas_segm_path, output_dir=dir_res_t)
    # Get the Jacobian matrix (not used now)
    jacobian_matrix_path = tr.jacobian_matrix(output_dir=dir_res_t)
    
    # Get the Jacobian determinant (not used now)
    jacobian_determinant_path = tr.jacobian_determinant(output_dir=dir_res_t)
    
    # Get the full deformation field (not used now)
    deformation_field_path = tr.deformation_field(output_dir=dir_res_t)
    deform_field = imageio.imread(deformation_field_path)
    
    transformed_atlas = imageio.imread(transformed_atlas_path)
    return transformed_atlas, transformed_atlas_path

segm_img, transformed_atlas_path = apply_transform(dir_res, moving_image_path, atlas_segm_path)

orig = imageio.imread(atlas_segm_path)
gt = imageio.imread(ground_truth_path)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)

ax0.imshow(orig[30], cmap='gray')
ax0.set_title('Original (atlas) mask')
ax1.imshow(segm_img[30], cmap='gray')
ax1.set_title('Transformed (atlas) mask')
ax2.imshow(gt[30], cmap='gray')
ax2.set_title('Ground Truth')


