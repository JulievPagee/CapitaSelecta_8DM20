# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:11:19 2021

@author: 20161879
"""

import numpy as np
import os
import cv2
import SimpleITK as sitk
import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import mahotas as mt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
import pandas as pd


def calc_haralick(roi):
    feature_vec = []

    texture_features = mt.features.haralick(roi)
    mean_ht = texture_features.mean(axis=0)

    [feature_vec.append(i) for i in mean_ht[0:9]]

    return np.array(feature_vec)


def harlick_features(img, h_neigh, ss_idx):
    print('[INFO] Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    if len(ss_idx) == 0:
        bar = progressbar.ProgressBar(maxval=len(patches), \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    else:
        bar = progressbar.ProgressBar(maxval=len(ss_idx), \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    if len(ss_idx) == 0:
        for i, p in enumerate(patches):
            bar.update(i + 1)
            h_features.append(calc_haralick(p))
    else:
        for i, p in enumerate(patches[ss_idx]):
            bar.update(i + 1)
            h_features.append(calc_haralick(p))

    # h_features = [calc_haralick(p) for p in patches[ss_idx]]

    return np.array(h_features)


def create_binary_pattern(img, p, r):
    print('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp - np.min(lbp)) / (np.max(lbp) - np.min(lbp)) * 255


def variance_feature_oud(img):
    print('[INFO] Computing variance feature.')
    img_array = np.array(img, dtype='float64')
    print(img_array.shape)
    lbl, nlbl = ndimage.label(img_array)
    var = ndimage.variance(img_array, labels=None, index=np.arange(1, nlbl + 1))
    return var
def subsample_idx(low, high, sample_size):
    return np.random.randint(low, high, sample_size)

def variance_feature(img_grey):
    print('[INFO] Computing variance feature.')
    varianceMatrix = ndimage.generic_filter(img_grey, np.var, size=1)
    return varianceMatrix


def gaussian_blur(img_gray, sigma=2):
    print('[INFO] Computing gaussian blur feature.')
    img_blurred = ndimage.gaussian_filter(img_gray, sigma=sigma)
    return img_blurred


def edges(img_grey):
    print('[INFO] Computing edges feature.')
    canny = feature.canny(img_grey, sigma=3)
    return canny


def create_features(img, img_gray, label, train=True):
    lbp_radius = 24  # local binary pattern neighbourhood
    h_neigh = 11  # haralick neighbourhood
    num_examples = 1000  # number of examples per image to use for training modelpip instal

    lbp_points = lbp_radius * 8
    h_ind = int((h_neigh - 1) / 2)
    feature_img = np.zeros((img.shape[0], img.shape[1], 7))
    feature_img[:, :, :3] = img
    feature_img[:, :, 3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    feature_img[:, :, 4] = variance_feature(img_gray)
    feature_img[:, :, 5] = gaussian_blur(img_gray)
    feature_img[:, :, 6] = edges(img_gray)
    img = None
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(feature_img.shape[0] * feature_img.shape[1], feature_img.shape[2])


    ss_idx = subsample_idx(0, features.shape[0], num_examples)
    features = features[ss_idx]


    h_features = harlick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    if train == True:

        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0] * label.shape[1], 1)
        labels = labels[ss_idx]
    else:
        labels = None

    return features, labels
def create_features2(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, _ = create_features(img, img_gray, label=None, train=False)

    return features
def compute_prediction(img, save_f, use_saved_f, patient, FEATURE_SAVE_PATH):
    slc = 0
    for i in img[0:86]:
        image = cv2.imread(path+i, 1)

        #image = np.mean(image, axis=2)
        image_shape = image.shape
        border = 5 # (haralick neighbourhood - 1) / 2
    
        img = cv2.copyMakeBorder(image, top=border, bottom=border, \
                                      left=border, right=border, \
                                      borderType = cv2.BORDER_CONSTANT, \
                                      value=[0, 0, 0])
        print(img.shape)
    
        features_savefile = patient+'_features.npy'
        features_save = os.path.join(FEATURE_SAVE_PATH, features_savefile)
        if slc == 0:
            features1 = create_features2(img)
        else:
            features = create_features2(img)
            features1 = np.concatenate((features1, features), axis=0)
        
        slc +=1
        
    np.save(features_save, features1)
            
    return features1

path = 'E:/CSMIA/Codes_def/ClassData/Validation/Slices/'
img_list = os.listdir(path)

f = compute_prediction(img_list,True, False, 'p119', 'E:/CSMIA/Codes_def/')