# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:37:07 2021

@author: s169369
"""
import elastix
import os
import imageio
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cv2 import *
import numpy as np
from scrollview import ScrollView

import rawpy
import imageio


ELASTIX_PATH = os.path.join(r'C:\Users\s169369\Documents\studie\2020-2021\05-03\Capita Selecta\Project\elastix.exe') 
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s169369\Documents\studie\2020-2021\05-03\Capita Selecta\Project\transformix.exe')


if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if non exists
if os.path.exists('results') is False:
    os.mkdir('results')
    
    
fixed_path = r'TrainingData\p102\mr_bffe.mhd'     
moving_path = r'TrainingData\p135\mr_bffe.mhd'
    
fixed_image = sitk.ReadImage(fixed_path)
fixed_im_array = sitk.GetArrayFromImage(fixed_image)

moving_image = sitk.ReadImage(moving_path)
moving_im_array = sitk.GetArrayFromImage(moving_image)


parameter_file_path = r'TrainingData\parameters.txt'

def Registration(fixed_image_path, atlas_path, ELASTIX_PATH, pnr ):
    "Function does registration for one atlas with one fixed image"
    
    parameter_file_path = r'TrainingData\parameters.txt'
    
    # Make a results directory if non exists
    if os.path.exists('results_{}'.format(pnr)) is False:
        os.mkdir('results_{}'.format(pnr))
    
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    el.register(
        fixed_image=fixed_image_path,
        moving_image=atlas_path,
        parameters=[parameter_file_path],
        output_dir='results_{}'.format(pnr))


# Perform registration 5 times
Registration(fixed_path, moving_path, ELASTIX_PATH, 'p102')




# # result jacobian:
# tr = elastix.TransformixInterface(parameters='res_chest/TransformParameters.0.txt',transformix_path=TRANSFORMIX_PATH)
# tr.jacobian_determinant('res_chest_p')
# jac_path = os.path.join('res_chest_p', 'spatialJacobian.mhd')
# jacobian = sitk.ReadImage(jac_path)
# jac_res = sitk.GetArrayFromImage(jacobian)
# jac_bin = jac_res > 0
# # now black spots in image are where folding occurs (negative jacobian)
# plt.figure()
# plt.imshow(jac_bin, cmap='gray')
# plt.title('Jacobian. black is negative = folding')

# # results image :
# result_path = os.path.join('res_chest_p', 'result.0.mhd')
# transformed_moving_image = sitk.ReadImage(result_path)
# tr_mov_im_array = sitk.GetArrayFromImage(transformed_moving_image)

# plt.figure()
# plt.imshow(tr_mov_im_array, cmap='gray')
# plt.title('transformed moving image')
# # Iteration_file_path_0 = 'res_ssp2/IterationInfo.0.R0.txt'
# # log0 = elastix.logfile(Iteration_file_path_0)

# # Iteration_file_path_1 = 'res_ssp2/IterationInfo.0.R1.txt'
# # log1 = elastix.logfile(Iteration_file_path_1)

# # Iteration_file_path_2 = 'res_ssp2/IterationInfo.0.R2.txt'
# # log2 = elastix.logfile(Iteration_file_path_2)

# # Iteration_file_path_3 = 'res_ssp2/IterationInfo.0.R3.txt'
# # log3 = elastix.logfile(Iteration_file_path_3)
# # Iteration_file_path_4 = 'res_ssp2/IterationInfo.0.R4.txt'
# # log4 = elastix.logfile(Iteration_file_path_4)

# # plt.figure()
# # plt.plot(log0['itnr'], log0['metric'], label = 'R0')
# # plt.plot(log1['itnr'], log1['metric'], label ='R1')
# # plt.plot(log2['itnr'], log2['metric'], label = 'R2')
# # plt.plot(log3['itnr'], log3['metric'], label = 'R3')
# # plt.plot(log4['itnr'], log4['metric'], label = 'R4')

# # plt.title('cost-functions')
# # plt.legend()
