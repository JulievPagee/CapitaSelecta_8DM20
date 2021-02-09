# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:53:54 2021

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

fixed_image_path = 'E:/CSMIA/NormData/norm_img_0.mhd'
moving_image_path = 'E:/CSMIA/NormData/norm_img_1.mhd'

dir_res = 'E:/CSMIA/resultstest5/'

ELASTIX_PATH = os.path.join(r'elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'transformix.exe')

if os.path.exists(dir_res) is False:
    os.mkdir(dir_res)

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[os.path.join(r'parameters.txt')],
    output_dir=dir_res)


itk_image = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))[:,:,20]
image_array_s = sitk.GetArrayFromImage(itk_image)
f_image = sitk.ReadImage(fixed_image_path)[:,:,20]
fixed_image_s = sitk.GetArrayFromImage(f_image)
m_image = sitk.ReadImage(moving_image_path)[:,:,20]
moving_image_s = sitk.GetArrayFromImage(m_image)

f, (ax0, ax1, ax2,ax3) = plt.subplots(1, 4)
ax0.imshow(fixed_image_s, cmap='gray')
ax0.set_title('fixed')
ax1.imshow(moving_image_s, cmap='gray')
ax1.set_title('moving')
ax2.imshow(image_array_s, cmap='gray')
ax2.set_title('registered image')

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

ax3.imshow(imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))[40])

deform_field = imageio.imread(deformation_field_path)