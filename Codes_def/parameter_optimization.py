# Optimization of parameters
# BELANGRIJK: we doen nu nog mutual information berekenen op slice 20, niet de hele image!
from Reg_functions import *
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np

# Adjust to your directory!
ELASTIX_PATH = os.path.join(r'C:\Users\s169369\Documents\studie\2020-2021\05-03\Capita Selecta\Project\elastix.exe') 
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s169369\Documents\studie\2020-2021\05-03\Capita Selecta\Project\transformix.exe')

# set path to train and atlas images
path_train = 'NormData/Train/'
path_atlas = 'NormData/Atlas/'

# read in path of training and atlas images (only mhd files):
Training_images_paths = []
Training_labels = ['p102', 'p109', 'p115', 'p117', 'p120', 'p128', 'p129']
for dirName, subdirList, fileList in os.walk(path_train):
    for filename in fileList:
        if '.mhd' in filename:
            Training_images_paths.append(os.path.join(dirName,filename))

Atlas_images_paths = []
Atlas_labels = ['p107', 'p108', 'p116', 'p127', 'p135']
for dirName, subdirList, fileList in os.walk(path_atlas):
    for filename in fileList:
        if '.mhd' in filename:
            Atlas_images_paths.append(os.path.join(dirName,filename))

# get path to directory where parameter files are stored
parameter_file_dir = 'Parameter_files'
# get path to the parameter file for nonrigid registration (actual registration parameter file)
parameter_file_path = 'Parameter_files/parameters.txt'

# Initialization methods to be tested
I_methods = ['none', 'rigid', 'affine', 'rigid_affine']

# Test to adjust penalty term with weights 0.1 to 3
penalty_weights = [0.005, 0.05, 0.5, 5]

# Test to adjust the parameter file with resolutions 1 to 6
ResValues = [1,2,3,4,5,6]

# Test to adjust parameter file with finest resolution values 4-64
FR_values = [4, 8, 16, 32, 64]

# define list with patients to bet tested (train data) = fixed images
pnr_list = Training_labels[0:1] # you can select a few training images here if you like

atlas_selection = Atlas_labels[0] # give here selection of atlasses used if you want

# HERE all training images are tested at one atlas ([0] FOR NOW):
# moving image = atlas image normally, but HERE it is a training image ONLY for parameter optimization
moving_path = Atlas_images_paths[0]

# fixed images = the training images. 

# Determine the best initialization step
best_initialization_idx = np.zeros(len(pnr_list),dtype = int)
best_initialization_method = []
for idx in range(len(pnr_list)):
    pnr = pnr_list[idx]
    fixed_path = Training_images_paths[idx] 
    best_initialization_idx[idx] = bestInitialization(I_methods, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_dir)
    max_ind = best_initialization_idx[idx]
    best_initialization_method.append(I_methods[max_ind]) 
# Majority voting nog niet gelukt omdat output dtype str heeft.
print(best_initialization_method)

# Determine the best weight of penalty term with one atlas on all 7 train images
best_penalty_weight = np.zeros(len(pnr_list), dtype=float)
for idx in range(len(pnr_list)):
    pnr = pnr_list[idx]
    fixed_path = Training_images_paths[idx]
    best_penalty_weight[idx] = bestPenalty(penalty_weights, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
# Majority voting nog niet gelukt omdat output dtype float heeft.
print(best_penalty_weight)

# Determine the best resolution with one atlas on all 7 train images
best_res_patient = np.zeros(len(pnr_list), dtype=int)
for idx in range(len(pnr_list)):
    pnr = pnr_list[idx]
    fixed_path = Training_images_paths[idx]
    best_res_patient[idx] = bestResolution(ResValues, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
# Majority voting
final_res = np.bincount(best_res_patient).argmax()
print(best_res_patient)

# Determine the best value for finest resolution
best_finest_resolution = np.zeros(len(pnr_list), dtype=int)
for idx in range(len(pnr_list)):
    pnr = pnr_list[idx]
    fixed_path = Training_images_paths[idx]
    best_finest_resolution[idx] = bestFinestResolution(FR_values, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
# Majority voting
final_finest_resolution = np.bincount(best_finest_resolution).argmax()
print(best_finest_resolution)
    
print("This is the end of parameter optimalization. See optimal values for the parameters in txt file.")

# Make txt file with best parameters
# NOTE: CHECK IF TEXT IS IN OUTPUT_FILE. iF NOT, YOU'VE TO RUN THIS PART AGAIN TO SEE TEXT IN TXT FILE. 
parameter_resultfile = open('output_file.txt', 'w+')
l_explain = ('This file shows the optimal parameters, which leads to the best registration (highest value for Mutual Information).\n')
l_init = ('The optimal initialization method before nonrigid registration is {} \n'.format(best_initialization_method))
l_penalty = ('The optimal weight for the penalty term to use is {} \n'.format(best_penalty_weight))
l_resolution = ('For optimal resolution, use  NumberOfResolutions: {} \n'.format(final_res))
l_finest_res = ('For optimal resolution, use Finest resolution value: {} \n').format(final_finest_resolution)
Lines = [l_explain, l_init, l_penalty, l_resolution, l_finest_res]
parameter_resultfile.writelines(Lines)


