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

# get path to parameter file
parameter_file_path = 'parameters.txt'

# Test to adjust penalty term with weights 0.1 to 3
penalty_weights = [0.1, 0.5, 1, 5]

# Test to adjust the parameter file with resolutions 1 to 6
ResValues = [1,2,3] #[1,2,3,4,5,6]

# define list with patients to bet tested (train data)
pnr_list = Training_labels[0:3] # you can select a few training images here if you like

# define fixed and moving image
moving_path = Atlas_images_paths[0]
# fixed image has to be the training images. 
# all training images are tested at one atlas:
for p_idx, pnr in enumerate(pnr_list):
    fixed_path = Training_images_paths[p_idx] 
    
    # Determine the best initialization step
    best_initialization = np.zeros(len(pnr_list))
    for idx in range(len(pnr_list)):
        pnr = pnr_list[idx]
        best_initialization[idx] = bestInitialization(I_methods, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
    print(best_initialization)

    # Determine the best weight of penalty term with one atlas on all 7 train images
    best_penalty_weight = np.zeros(len(pnr_list), dtype=float)
    for idx in range(len(pnr_list)):
        pnr = pnr_list[idx]
        best_penalty_weight[idx] = bestPenalty(penalty_weights, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
    print(best_penalty_weight)
    
    # Determine the best resolution with one atlas on all 7 train images
    best_res_patient = np.zeros(len(pnr_list), dtype=int)
    for idx in range(len(pnr_list)):
        pnr = pnr_list[idx]
        best_res_patient[idx] = bestResolution(ResValues, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
    print(best_res_patient)
    
    # Majority voting
    final_res = np.bincount(best_res_patient).argmax()
    print(final_res)
    print("done")

# To DO: Make txt file with best parameters
parameter_resultfile = open('output_file.txt', 'w+')
l_explain = ('This file shows the optimal parameters, which leads to the best registration (highest value for Mutual Information.\n')
l_init = ('The optimal initialization method before nonrigid registration is {} \n'.format(best_initialization))
l_penalty = ('The optimal weight for the penalty term to use is {} \n'.format(best_penalty_weight))
l_resolution = ('For optimal resolution, use  NumberOfResolutions: {} \n'.format(best_res_patient))
Lines = [l_explain, l_init, l_penalty, l_resolution]
parameter_resultfile.writelines(Lines)


