# -*- coding: utf-8 -*-
"""
Created on 12/03/2021

@author: 20165272
"""
import cv2
import numpy as np
import SimpleITK as sitk

import os


#option 1
#ref: https://stackoverflow.com/questions/31573872/how-to-read-multiple-images-and-create-a-3d-matrix-with-them
#arrays = []
#for number in range(0, 299):
#    numstr = str(number).zfill(3)
 #   fname = numstr + '.bmp'
 #   a = imread(fname, flatten=1)
 #   arrays.append(a)
#data = np.array(arrays)
#dit lijkt me vrij omslachtig, np.concatenate doet ook de job.

#option 2
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.dstack.html
#a = figure1   #np.array format
#b = figure2   #np.array format
#np.concatenate((a, b), axis=0)  #stackt ze in verticale richting (links =1, rechts =2)
#np.concatenate((a, b.), axis=1) #stackt ze in horizontale richting
#np.concatenate((a,b) axis=2)    #axis 2 obv bovenste redenatie
#option 3
#https://nilearn.github.io/modules/generated/nilearn.image.concat_imgs.html



#doel (333,271,86) array maken.
#option 2 lijkt me het handiste
def slice_to_3D(data_list, save_dir, save_name):

    final_image = np.array([])
    final_image = np.zeros((86,333,271))
    firstIteration = True
    for item in data_list:
        img = data_path+ patient_path + item + '.png'
        img = sitk.ReadImage(img)
        img = sitk.GetArrayFromImage(img)
        img = np.expand_dims(img, axis=0)

        if firstIteration:
            final_image = img
            firstIteration=False
        else:
            final_image = np.concatenate(([final_image, img ]), axis=0)


    final_image = sitk.GetImageFromArray(final_image)
    save_path = os.path.join(save_dir, save_name)  # get the save path
    sitk.WriteImage(final_image, save_path)

patient_num = '107'
patient_path = 'p' + patient_num +'_lb_'
data_path = 'E:/CSMIA/2D_ML/lb/'
data_list = list(range(0,86))
data_list = [str(i) for i in data_list]
save_dir= 'E:/CSMIA/2D_ML/'
save_name= 'final_image.mhd'

slice_to_3D(data_list, save_dir, save_name)

