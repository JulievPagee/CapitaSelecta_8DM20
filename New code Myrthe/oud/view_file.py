# -*- coding: utf-8 -*-
"""


@author: 20161879

Description:
    Code to load and view the fixed, moving and registration.

"""
import matplotlib.pyplot as plt
import os
import torch
import SimpleITK as sitk

dir_res = 'E:/CSMIA/resultstest5/'
#Load fixed, moving and registration to plot with scroll in Z-direction.
fixed_path = 'E:/CSMIA/NormData/norm_img_0.mhd' 
fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path, sitk.sitkFloat32))
fixed = torch.from_numpy(fixed).permute(1,2,0)

moving_path = 'E:/CSMIA/NormData/norm_img_1.mhd'
moving = sitk.GetArrayFromImage(sitk.ReadImage(moving_path))
moving = torch.from_numpy(moving).permute(1,2,0)


reg_path = os.path.join(dir_res, 'result.0.mhd')
reg = sitk.GetArrayFromImage(sitk.ReadImage(reg_path))
reg = torch.from_numpy(reg).permute(1,2,0)


class IndexTracker(object):
    def __init__(self, ax, fixed, FL, reg):
        
        self.ax0 = ax[0]
        ax[0].set_title('fixed')
        
        self.ax1 = ax[1]
        ax[1].set_title('moving')
        
        
        self.ax2 = ax[2]
        ax[2].set_title('Registered')
        
        
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

tracker = IndexTracker(ax, fixed, moving, reg)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
