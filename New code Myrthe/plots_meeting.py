# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:55:47 2021

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

######### Functie overslaan ###############
class IndexTracker(object):
    def __init__(self, ax, fixed, FL, reg, fixed_i, FL_i, reg_i):
        
        self.ax0 = ax[0,0]
        ax[0,0].set_title('Atlas segmentation')
        self.ax1 = ax[0,1]
        ax[0,1].set_title('Predicted segmentation')
        self.ax2 = ax[0,2]
        ax[0,2].set_title('True segmentation')
 
        self.ax3 = ax[1,0]
        ax[1,0].set_title('Atlas image')
        self.ax4 = ax[1,1]
        ax[1,1].set_title('Registered image')
        self.ax5 = ax[1,2]
        ax[1,2].set_title('True image')       
        
        self.fixed = fixed
        rows0, cols0, self.slices0 = fixed.shape
        self.ind0 = self.slices0//2
        self.FL = FL
        rows1, cols1, self.slices1 = FL.shape
        self.ind1 = self.slices1//2
        self.reg = reg
        rows2, cols2, self.slices2 = reg.shape
        self.ind2 = self.slices2//2
        
        self.fixed_i = fixed_i
        rows3, cols3, self.slices3 = fixed.shape
        self.ind3 = self.slices3//2
        self.FL_i = FL_i
        rows4, cols4, self.slices4 = FL.shape
        self.ind4 = self.slices4//2
        self.reg_i = reg_i
        rows5, cols5, self.slices5 = reg.shape
        self.ind5 = self.slices5//2


        self.im0 = ax[0,0].imshow(self.fixed[:, :, self.ind0], cmap='gray')        
        self.im1 = ax[0,1].imshow(self.FL[:, :, self.ind1], cmap='gray')       
        self.im2 = ax[0,2].imshow(self.reg[:, :, self.ind2], cmap='gray')
        
        self.im3 = ax[1,0].imshow(self.fixed_i[:, :, self.ind3], cmap='gray')        
        self.im4 = ax[1,1].imshow(self.FL_i[:, :, self.ind4], cmap='gray')       
        self.im5 = ax[1,2].imshow(self.reg_i[:, :, self.ind5], cmap='gray')
        
        self.update()
                 
    def onscroll(self, event):
        #regint("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind0 = (self.ind0 + 2) % self.slices0
            self.ind1 = (self.ind1 + 2) % self.slices1
            self.ind2 = (self.ind2 + 2) % self.slices2
            self.ind3 = (self.ind3 + 2) % self.slices3
            self.ind4 = (self.ind4 + 2) % self.slices4
            self.ind5 = (self.ind5 + 2) % self.slices5
    
        else:
            self.ind0 = (self.ind0 - 2) % self.slices0
            self.ind1 = (self.ind1 - 2) % self.slices1
            self.ind2 = (self.ind2 - 2) % self.slices2
            self.ind3 = (self.ind3 - 2) % self.slices3
            self.ind4 = (self.ind4 - 2) % self.slices4
            self.ind5 = (self.ind5 - 2) % self.slices5
        self.update()

    def update(self):
        self.im0.set_data(self.fixed[:, :, self.ind0])
        ax[0,0].set_ylabel('slice %s' % self.ind0)
        self.im0.axes.figure.canvas.draw()
        self.im1.set_data(self.FL[:, :, self.ind1])
        ax[0,1].set_ylabel('slice %s' % self.ind1)
        self.im1.axes.figure.canvas.draw()
        self.im2.set_data(self.reg[:, :, self.ind2])
        ax[0,2].set_ylabel('slice %s' % self.ind2)
        self.im2.axes.figure.canvas.draw()

        self.im3.set_data(self.fixed_i[:, :, self.ind3])
        ax[1,0].set_ylabel('slice %s' % self.ind3)
        self.im3.axes.figure.canvas.draw()
        self.im4.set_data(self.FL_i[:, :, self.ind4])
        ax[1,1].set_ylabel('slice %s' % self.ind4)
        self.im4.axes.figure.canvas.draw()
        self.im5.set_data(self.reg_i[:, :, self.ind5])
        ax[1,2].set_ylabel('slice %s' % self.ind5)
        self.im5.axes.figure.canvas.draw()

################ Segmentaties ######################

############ aanpassen ##################
codes_def_path = 'E:/CSMIA/Codes_def/'
DATA_DIR = codes_def_path+'TrainingData/'
img = 1
############ aanpassen ##################

fig, ax = plt.subplots(2, 3)
img_atlas = [['p133', 'p127'], ['p119', 'p107'], ['p125', 'p127']]
segm_images = [codes_def_path+'Results_atl_reg/transform_p133/Transformed_masks/mask_B_spline_p133\\result.mhd', codes_def_path+'Results_atl_reg/transform_p119/Transformed_masks/mask_B_spline_p119\\result.mhd', codes_def_path+'Results_atl_reg/transform_p125/Transformed_masks/mask_B_spline_p125\\result.mhd']

#create images
atlas_seg_path = DATA_DIR+img_atlas[img][1]+'/prostaat.mhd'
atlas_seg = imageio.imread(atlas_seg_path)
atlas_seg= torch.from_numpy(atlas_seg).permute(1,2,0)

pred_seg_path = segm_images[img]
pred_seg = imageio.imread(pred_seg_path)
pred_seg = torch.from_numpy(pred_seg).permute(1,2,0)

gt_seg_path = DATA_DIR+img_atlas[img][0]+'/prostaat.mhd'
gt_seg = imageio.imread(gt_seg_path)
gt_seg = torch.from_numpy(gt_seg).permute(1,2,0)

DATA_DIR_i = codes_def_path+'NormData/'

#create images
atlas_path = DATA_DIR_i+'Atlas/norm_img_{}.mhd'.format(img_atlas[img][1])
atlas = imageio.imread(atlas_path)
atlas= torch.from_numpy(atlas).permute(1,2,0)

pred_path = codes_def_path + 'Results_atl_reg/results_B_spline_{}/atlas_{}/result.0.mhd'.format(img_atlas[img][0],img_atlas[img][1])
pred = imageio.imread(pred_path)
pred = torch.from_numpy(pred).permute(1,2,0)

gt_path = DATA_DIR_i+'Validation/norm_img_{}.mhd'.format(img_atlas[img][0])
gt = imageio.imread(gt_path)
gt = torch.from_numpy(gt).permute(1,2,0)


tracker = IndexTracker(ax, atlas_seg, pred_seg, gt_seg, atlas, pred, gt)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()




