# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:39:57 2021

@author: 20161879
"""
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

def read_data_labelled(image_dir, label_dir, b_n, e_n):

    print ('[INFO] Reading labelled image data.')

    filelist = glob(os.path.join(image_dir, '*.png'))[b_n:e_n]
    labellist = glob(os.path.join(label_dir, '*.png'))[b_n:e_n]
    image_list = []
    label_list = []

    for idx, file in enumerate(filelist):

        image_list.append(cv2.imread(file, 1))
        label_list.append(cv2.imread(labellist[idx], 0))

    return image_list, label_list

def read_data_unlabelled(image_dir, b_n, e_n):
    print('[INFO] Reading unlabelled image data')
    filelist = glob(os.path.join(image_dir, '*.png'))[b_n:e_n]
    image_list = []
    for idx, file in enumerate(filelist):
        image_list.append(cv2.imread(file, 1))
    return image_list

def subsample(features, labels, low, high, sample_size):

    idx = np.random.randint(low, high, sample_size)

    return features[idx], labels[idx]

def subsample_idx(low, high, sample_size):

    return np.random.randint(low,high,sample_size)

def calc_haralick(roi):

    feature_vec = []

    texture_features = mt.features.haralick(roi)
    mean_ht = texture_features.mean(axis=0)

    [feature_vec.append(i) for i in mean_ht[0:9]]

    return np.array(feature_vec)

def harlick_features(img, h_neigh, ss_idx):

    print ('[INFO] Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    if len(ss_idx) == 0 :
        bar = progressbar.ProgressBar(maxval=len(patches), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    else:
        bar = progressbar.ProgressBar(maxval=len(ss_idx), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    if len(ss_idx) == 0:
        for i, p in enumerate(patches):
            bar.update(i+1)
            h_features.append(calc_haralick(p))
    else:
        for i, p in enumerate(patches[ss_idx]):
            bar.update(i+1)
            h_features.append(calc_haralick(p))

    #h_features = [calc_haralick(p) for p in patches[ss_idx]]

    return np.array(h_features)

def create_binary_pattern(img, p, r):

    print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    output = (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def variance_feature_oud(img):
    print('[INFO] Computing variance feature.')
    img_array = np.array(img, dtype='float64')
    print(img_array.shape)
    lbl, nlbl = ndimage.label(img_array)
    var = ndimage.variance(img_array, labels=None, index=np.arange(1, nlbl + 1))
    return var

def variance_feature(img):
    print('[INFO] Computing variance feature.')
    varianceMatrix = ndimage.generic_filter(img, np.var, size=1)
    result = np.mean(varianceMatrix, axis=2)
    return result

def gaussian_blur(img, sigma=2):
    print('[INFO] Computing gaussian blur feature.')
    img_array = np.array(img)
    img_blurred = ndimage.gaussian_filter(img_array, sigma=sigma)
    result = np.mean(img_blurred, axis=2)
    return result

def edges(img):
    print('[INFO] Computing edges feature.')
    img_array = np.array(img)
    img_array = np.mean(img_array, axis=2)
    canny = feature.canny(img_array, sigma=3)
    return canny

def create_features(img, img_gray, label, train=True):

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 1000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)
    feature_img = np.zeros((img.shape[0],img.shape[1],7))
    feature_img[:,:,:3] = img
    feature_img[:,:,3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    feature_img[:,:,4] = variance_feature(img)
    feature_img[:,:,5] = gaussian_blur(img)
    feature_img[:,:,6] = edges(img)
    img = None
    feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    if train == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx = []

    h_features = harlick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    if train == True:

        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        labels = labels[ss_idx]
    else:
        labels = None

    return features, labels

def create_training_dataset(image_list, label_list):

    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

    X = []
    y = []

    for i, img in enumerate(image_list):
        print('[INFO] Calculating feature for training image: %d' %i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, label_list[i])
        X.append(features)
        y.append(labels)

    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print ('[INFO] Feature vector size:', X_train.shape)

    return X_train, X_test, y_train, y_test

def create_unlabelled_dataset(image_list):
    print('[INFO] Creating unlabelled dataset on %d images.' %len(image_list))
    X = []

    for i, img in enumerate(image_list):
        print('[INFO] Creating features for unlabelled image: %d' %i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, None, train=False)
        X.append(features)
    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])

    return X

def train_model(X, y, classifier, fr):

    if classifier == "SVM":
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        if fr:
            print('SVM with PCA')
            steps = [('pca', PCA(n_components=10)), ('m', SVC(gamma='scale'))]
            model = Pipeline(steps=steps)
        else:
            print('SVM')
            model = SVC(gamma='scale')
        model.fit(X, y)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)
    dce = dice(pred,y)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('[RESULTS] Dice: %.2f' %dce)
    print ('--------------------------------')
    return pred

def main(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled, classifier, fr, output_model):

    start = time.time()

    image_list_labelled, label_list_labelled = read_data_labelled(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled)
    image_list_unlabelled = read_data_unlabelled(image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled)
    X_train, X_test, y_train, y_test = create_training_dataset(image_list_labelled, label_list_labelled)
    X_unlabelled = create_unlabelled_dataset(image_list_unlabelled)
    model = train_model(X_train, y_train, classifier, fr)
    pred = test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))
    print ('Processing time:',time.time()-start)
    
    return pred