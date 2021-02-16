# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:37:07 2021

@author: s169369
"""

import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from cv2 import * # Used for GridSearch
# Functions used during registration:
from Reg_functions import *

# Adjust to your directory!
ELASTIX_PATH = os.path.join("C:/Users/20175722/Documents/Master jaar 1/Kwartiel 3/8DM20 - CS/elastix-5.0.0-win64/elastix.exe") 
TRANSFORMIX_PATH = os.path.join("C:/Users/20175722/Documents/Master jaar 1/Kwartiel 3/8DM20 - CS/elastix-5.0.0-win64/transformix.exe")
fixed_path = 'NormData/norm_img_p102.mhd'     
moving_path = 'NormData/norm_img_p107.mhd'
parameter_file_path = 'New code Myrthe/parameters.txt'

# Give error when Elastix or Transformix are not found
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if none exists
if os.path.exists('Results') is False:
    os.mkdir('Results')
    
# Read in images    
fixed_image = sitk.ReadImage(fixed_path)
fixed_im_array = sitk.GetArrayFromImage(fixed_image)
moving_image = sitk.ReadImage(moving_path)
moving_im_array = sitk.GetArrayFromImage(moving_image)

# Perform registration with Registration from Reg_functions
dir_res = Registration(fixed_path, moving_path, parameter_file_path, ELASTIX_PATH, 'p102')

# Code Myrthe for plotting
itk_image = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))[:,:,20]
image_array_s = sitk.GetArrayFromImage(itk_image)
f_image = sitk.ReadImage(fixed_path)[:,:,20]
fixed_image_s = sitk.GetArrayFromImage(f_image)
m_image = sitk.ReadImage(moving_path)[:,:,20]
moving_image_s = sitk.GetArrayFromImage(m_image)

f, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(fixed_image_s, cmap='gray')
ax0.set_title('fixed')
ax1.imshow(moving_image_s, cmap='gray')
ax1.set_title('moving')
ax2.imshow(image_array_s, cmap='gray')
ax2.set_title('registered image')
plt.show()

# Calculate mutual information with function from Reg_function
mutual_info = mutual_information(fixed_image_s, image_array_s)
print(mutual_info)

# Know that the code is done running
print("End")
