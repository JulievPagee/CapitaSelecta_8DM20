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
import pandas as pd

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
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def variance_feature_oud(img):
    print('[INFO] Computing variance feature.')
    img_array = np.array(img, dtype='float64')
    print(img_array.shape)
    lbl, nlbl = ndimage.label(img_array)
    var = ndimage.variance(img_array, labels=None, index=np.arange(1, nlbl + 1))
    return var

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

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 1000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)
    feature_img = np.zeros((img.shape[0],img.shape[1],7))
    feature_img[:,:,:3] = img
    feature_img[:,:,3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    feature_img[:,:,4] = variance_feature(img_gray)
    feature_img[:,:,5] = gaussian_blur(img_gray)
    feature_img[:,:,6] = edges(img_gray)
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

def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')
    return pred

def main(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, x_test_save, y_test_save, unlabelled_save, save = True, use_saved = True):

    start = time.time()

    image_list_labelled, label_list_labelled = read_data_labelled(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled)
    image_list_unlabelled = read_data_unlabelled(image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled)

    if use_saved:
        X_train = np.load(x_train_save)
        y_train = np.load(y_train_save)
        X_test = np.load(x_test_save)
        y_test = np.load(y_test_save)
        # X_unlabelled = np.load(unlabelled_save)
        print('Features loaded from file')
    else:
        X_train, X_test, y_train, y_test = create_training_dataset(image_list_labelled, label_list_labelled)
        # X_unlabelled = create_unlabelled_dataset(image_list_unlabelled)
        print('Feature extraction time in minutes:', (time.time()-start)/60)

        if save:
            np.save(x_train_save, X_train)
            np.save(y_train_save, y_train)
            np.save(x_test_save, X_test)
            np.save(y_test_save, y_test)
            # np.save(unlabelled_save, X_unlabelled)

    model = train_model(X_train, y_train, classifier, fr)
    pred = test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))
    print ('Total processing time in minutes:',(time.time()-start)/60)
    print('Model saved as:', output_model)
    
    return pred

def SSL(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, x_test_save, y_test_save, unlabelled_save, save = True, use_saved = True):
    start = time.time()

    image_list_labelled, label_list_labelled = read_data_labelled(image_dir_labelled, label_dir, b_n_labelled,
                                                                  e_n_labelled)
    image_list_unlabelled = read_data_unlabelled(image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled)

    if use_saved:
        X_train = np.load(x_train_save)
        y_train = np.load(y_train_save)
        X_test = np.load(x_test_save)
        y_test = np.load(y_test_save)
        X_unlabelled = np.load(unlabelled_save)
        print('Features loaded from file')
    else:
        X_train, X_test, y_train, y_test = create_training_dataset(image_list_labelled, label_list_labelled)
        X_unlabelled = create_unlabelled_dataset(image_list_unlabelled)
        print('Feature extraction time in minutes:', (time.time() - start) / 60)

        if save:
            np.save(x_train_save, X_train)
            np.save(y_train_save, y_train)
            np.save(x_test_save, X_test)
            np.save(y_test_save, y_test)
            np.save(unlabelled_save, X_unlabelled)

    iterations = 0
    train_f1s = []
    test_f1s = []
    pseudo_labels = []

    # Assign value to initiate while loop
    high_prob = [1]

    while len(high_prob) > 0:
        model = train_model(X_train, y_train, classifier, fr)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        train_f1 = metrics.f1_score(y_train, y_hat_train)
        test_f1 = metrics.f1_score(y_test, y_hat_test)

        print(f"Iteration {iterations}")
        print(f"Train f1: {train_f1}")
        print(f"Test f1: {test_f1}")
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

        # Generate predictions for the unlabelled data
        print("Now predicting labels for unlabelled data")

        pred_probs = model.predict_proba(X_unlabelled)
        preds = model.predict(X_unlabelled)
        prob_0 = pred_probs[:,0]
        prob_1 = pred_probs[:,1]

        # Check if probabilities >0


        # Store predictions in pd dataframe
        df_pred_prob = pd.DataFrame([])
        df_pred_prob['preds'] = preds
        df_pred_prob['prob_0'] = prob_0
        df_pred_prob['prob_1'] = prob_1
        df_pred_prob.index = X_unlabelled.index

        # Separate predictions with > 99% probability
        high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99], df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]], axis=0)
        print(f"{len(high_prob)} high-probability predictions added to training data")

        pseudo_labels.append(len(high_prob))

        # Really add them to x_train and the labels to y_train
        X_train = pd.concat([X_train, X_unlabelled.loc[high_prob.index]], axis=0)
        y_train = pd.concat([y_train, high_prob.preds])

        # Drop pseudo labeled instances from the unlabelled data.
        X_unlabelled = X_unlabelled.drop(index=high_prob.index)
        print(f"{len(X_unlabelled)} unlabelled instances remaining. \n")

        #update iterations counter
        iterations += 1

    pkl.dump(model, open(output_model, "wb"))
    print('Total processing time in minutes:', (time.time() - start) / 60)
    print('Model saved as:', output_model)
