# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:38:17 2021

@author: 20161879
"""
import os
import SimpleITK as sitk 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
data_path = 'E:/CSMIA/Codes_def/TrainingData/'
data_folders = os.listdir(data_path)
save_dir = 'E:/CSMIA/2D_ML/'

if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    os.mkdir(save_dir+'im')
    os.mkdir(save_dir+'lb')

for idx, nr in enumerate(data_folders):
    im_path = data_path+nr+'/mr_bffe.mhd'
    lb_path = data_path+nr+'/prostaat.mhd'
    
    im = sitk.ReadImage(im_path)
    lb = sitk.ReadImage(lb_path)
    
    im = sitk.GetArrayFromImage(im)
    lb = sitk.GetArrayFromImage(lb)
    
    for i in range(86):
        im_s = im[i]
        lb_s = lb[i]
        
        im_s = (255.0 / im_s.max() * (im_s - im_s.min())).astype(np.uint8)
        lb_s = (255.0 / lb_s.max() * (lb_s - lb_s.min())).astype(np.uint8)
        image = Image.fromarray(im_s)
        label = Image.fromarray(lb_s)
        

        image.save(save_dir+'im/'+nr+'_im_'+str(i)+'.png')
        label.save(save_dir+'lb/'+nr+'_lb_'+str(i)+'.png')
