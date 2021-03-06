# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:03:18 2021

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
import torch


def mutual_information(fixed_img, atlas_img):
        """ Mutual information for joint histogram
        """
        hist_2d, x_edges, y_edges = np.histogram2d(fixed_img.ravel(), atlas_img.ravel(), bins=20)
        # Convert bins counts to probability values
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
#Functie die de path van de beste segmentatie teruggeeft.
def GetBestAtlasSegmentationFile(DATA_DIR, DATA_DIR_t, fixed_img_path, atlasses, Atlas_labels):
    #Mutual information berekenen
    MIlist = []
    for i in range(5):
        atlas_path = os.path.join(DATA_DIR_t, atlasses[i], 'result.0.mhd')
        fixed_img = imageio.imread(fixed_img_path)
        atlas_img = imageio.imread(atlas_path)
        
        MI = mutual_information(fixed_img, atlas_img)
        MIlist.append(MI)
    # Index vinden van de atlas met de hoogste mutual information
    index = MIlist.index(max(MIlist))
    
    #De binary file van de beste atlas inladen
    atlas_seg_path = os.path.join(DATA_DIR, Atlas_labels[index], 'prostaat.mhd')
    atlas_seg = imageio.imread(atlas_seg_path)
    
    return atlas_seg, atlas_seg_path, index
 
    
def new_param_file(par_path, changing_par, orig_val, new_val):
    # Open the parameter file to make transform binary
    with open(par_path) as f:
        par_file = f.readlines()
    # Search per line if the FinalBSplineInterpolationOrder is specified there
    line_nr = 0
    for line in par_file:
        if changing_par in line:
            #Change from the default to the new value 
            line = line.replace(orig_val, new_val) 
            #Adjust it in the file
            par_file[line_nr] = line 
        line_nr = line_nr + 1
        
    f.close()
    #create new file for the binary transform
    adj_par_path = par_path.replace(".txt", "_binary.txt")
    a_file = open(adj_par_path, "w")
    a_file.writelines(par_file)
    a_file.close()
    print('new file saved')
    return adj_par_path
def Registration(dir_res, fixed_image_path, moving_image_path, parameter_path, ELASTIX_PATH, pnr, anr, method, init):
    "Function to registrate one moving on one fixed image. pnr should be given as 'p102'"
    
    # Make a results folder if non exists (in Codes_def folder)
    
    if init:
        output = dir_res+'results_{}_{}/'.format(method, pnr)
        output_dir = output+'atlas_{}'.format(anr)
        if os.path.exists(output) is False:
            os.mkdir(output)
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        
    else:
        output = dir_res+'results_{}/'.format(pnr)
        output_dir = output+'atlas_{}'.format(anr)
        if os.path.exists(output) is False:
            os.mkdir(output)
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    el.register(
        fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=output_dir)
    return output_dir

def registration_func(pnr, anr, dir_res, ELASTIX_PATH, fixed_path, moving_path_i, parameter_file_dir):
    # first perform rigid registration
    parameters_init_r = '{}/parameters_rigid.txt'.format(parameter_file_dir)
    dir_res_init_reg = Registration(dir_res, fixed_path, moving_path_i, parameters_init_r, ELASTIX_PATH, pnr, anr, 'RA_rigid', True)
    
    # determine new moving path for affine registration
    moving_path_r = '{}/result.0.mhd'.format(dir_res_init_reg)
    
    # Registrate with nonrigid BSpline transform (actual registration)
    parameter_path = '{}/parameters_FINAL.txt'.format(parameter_file_dir)
    
    # Continue Bspline registration and load the images
    dir_res = Registration(dir_res, fixed_path, moving_path_r, parameter_path, ELASTIX_PATH, pnr, anr, 'B_spline', True)
    registered_image_path = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))
    image_registered = sitk.GetArrayFromImage(registered_image_path)
    
    return registered_image_path, image_registered


def Deformation(mask_segm_path, par_path, method, pnr, dir_res_i, TRANSFORMIX_PATH):
    dir_res = dir_res_i+"Transformed_masks/mask_{}_{}".format(method, pnr)
    if os.path.exists(dir_res_i+"Transformed_masks/") is False:
        os.mkdir(dir_res_i+"Transformed_masks/")
    if os.path.exists(dir_res_i+"Transformed_masks/mask_{}_{}".format(method, pnr)) is False:
        os.mkdir(dir_res_i+"Transformed_masks/mask_{}_{}".format(method, pnr))

    path = new_param_file(par_path, "FinalBSplineInterpolationOrder", '3', '0')
    # Make a new transformix object tr with the CORRECT PATH to transformix
    tr = elastix.TransformixInterface(parameters=path,
                                      transformix_path=TRANSFORMIX_PATH)

    # Transform the mask with the transformation parameters
    transformed_image_path = tr.transform_image(mask_segm_path, output_dir=dir_res)
 
    transf_mask = sitk.ReadImage(transformed_image_path)
    transf_mask  = sitk.GetArrayFromImage(transf_mask )
    
    # Get the Jacobian matrix (not yet used)
    jacobian_matrix_path = tr.jacobian_matrix(output_dir=dir_res)
    
    # Get the Jacobian determinant (not yet used)
    jacobian_determinant_path = tr.jacobian_determinant(output_dir=dir_res)
    
    # Get the full deformation field (not yet used)
    deformation_field_path = tr.deformation_field(output_dir=dir_res)

    return transf_mask, transformed_image_path 

#function with initializaiton by both rigidtransform
def deform_func(results_dir, pnr, atlas_seg_path, atlas_fold, dir_res, TRANSFORMIX_PATH):
    # first perform rigid registration
    moving_mask_path = atlas_seg_path
    parameters_init_r = results_dir+'results_RA_rigid_{}/{}/TransformParameters.0.txt'.format(pnr, atlas_fold)
    transf_mask_r, transformed_image_path_r = Deformation(moving_mask_path, parameters_init_r, 'RA_rigid', pnr, dir_res, TRANSFORMIX_PATH) 
        
    # determine new moving path for affine registration
    moving_path_r = transformed_image_path_r 
    
    # Registrate with nonrigid BSpline transform (actual registration)
    parameter_path = results_dir+'results_B_spline_{}/{}/TransformParameters.0.txt'.format(pnr, atlas_fold)
    
    # Continue Bspline registration and load the images
    transf_mask_b, transformed_image_path_b = Deformation(moving_path_r, parameter_path, 'B_spline', pnr, dir_res, TRANSFORMIX_PATH)

    
    return transformed_image_path_b, transf_mask_b



def plot_segm(atlas_seg, pred_seg, gt_seg):
       
        class IndexTracker(object):
            def __init__(self, ax, fixed, FL, reg):
                
                self.ax0 = ax[0]
                ax[0].set_title('Atlas segmentation')
                
                self.ax1 = ax[1]
                ax[1].set_title('Predicted segmentation')
                
                
                self.ax2 = ax[2]
                ax[2].set_title('True segmentation')
                
                
                self.fixed = fixed
                rows0, cols0, self.slices0 = fixed.shape
                self.ind0 = self.slices0//2
                
                self.FL = FL
                rows1, cols1, self.slices1 = FL.shape
                self.ind1 = self.slices1//2
                
                
                self.reg = reg
                rows3, cols3, self.slices3 = reg.shape
                self.ind3 = self.slices3//2
        
        
                self.im0 = ax[0].imshow(self.fixed[:, :, self.ind0], cmap='gray')
                
                self.im1 = ax[1].imshow(self.FL[:, :, self.ind1], cmap='gray')
                
                self.im2 = ax[2].imshow(self.reg[:, :, self.ind3], cmap='gray')
                
                
                self.update()
                         
            def onscroll(self, event):
                #regint("%s %s" % (event.button, event.step))
                if event.button == 'up':
                    self.ind0 = (self.ind0 + 1) % self.slices0
                    self.ind1 = (self.ind1 + 1) % self.slices1
        
                    self.ind3 = (self.ind3 + 1) % self.slices3
                else:
                    self.ind0 = (self.ind0 - 1) % self.slices0
                    self.ind1 = (self.ind1 - 1) % self.slices1
        
                    self.ind3 = (self.ind3 - 1) % self.slices3
                self.update()
        
            def update(self):
                self.im0.set_data(self.fixed[:, :, self.ind0])
                ax[0].set_ylabel('slice %s' % self.ind0)
                self.im0.axes.figure.canvas.draw()
                
                self.im1.set_data(self.FL[:, :, self.ind1])
                ax[1].set_ylabel('slice %s' % self.ind1)
                self.im1.axes.figure.canvas.draw()
                
                self.im2.set_data(self.reg[:, :, self.ind3])
                ax[2].set_ylabel('slice %s' % self.ind3)
                self.im2.axes.figure.canvas.draw()
        
        fig, ax = plt.subplots(1, 3)
        
        tracker = IndexTracker(ax, atlas_seg, pred_seg, gt_seg)
        
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

