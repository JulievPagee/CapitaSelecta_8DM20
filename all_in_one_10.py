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
    print('[INFO] Reading labelled image data.')

    filelist = glob(os.path.join(image_dir, '*.png'))[b_n:e_n]
    labellist = glob(os.path.join(label_dir, '*.png'))[b_n:e_n]
    image_list = []
    label_list = []

    for idx, file in enumerate(filelist):
        image_list.append(cv2.imread(file, 1))
        label_list.append(cv2.imread(labellist[idx], 0))
    print("Labelled data images: {}".format(len(image_list)))
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
    return np.random.randint(low, high, sample_size)


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


def create_features(img, img_gray, label, train=True, subsamplen=True):
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

    if subsamplen:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx=[]

    h_features = harlick_features(img_gray, h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    if train == True:

        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0] * label.shape[1], 1)
        labels = labels[ss_idx]
    else:
        labels = None

    return features, labels


def create_training_dataset(image_list, label_list, test=False):
    print('[INFO] Creating training dataset on %d image(s).' % len(image_list))

    X = []
    y = []

    for i, img in enumerate(image_list):
        print('[INFO] Calculating feature for training image: %d' % i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, label_list[i])
        X.append(features)
        y.append(labels)

    X = np.array(X)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y = np.array(y)
    y = y.reshape(y.shape[0] * y.shape[1], y.shape[2]).ravel()

    if test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        output = [X_train, X_test, y_train, y_test]
    else:
        X_train, y_train = X, y
        output = [X_train, y_train]

    print('[INFO] Feature vector size:', X_train.shape)

    return output

def create_unlabelled_dataset(image_list):
    print('[INFO] Creating unlabelled dataset on %d images.' % len(image_list))
    X = []

    for i, img in enumerate(image_list):
        print('[INFO] Creating features for unlabelled image: %d' % i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = create_features(img, img_gray, None, train=False)
        X.append(features)
    X = np.array(X)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])

    return X

def train_model(X, y, classifier, fr, SSL=True):
    if SSL:
        y = y.values.ravel()
    if classifier == "SVM":
        from sklearn.svm import SVC
        print('[INFO] Training Support Vector Machine model.')
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
        print('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)
    elif classifier == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)

    print('[INFO] Model training complete.')
    print('[INFO] Training Accuracy: %.2f' % model.score(X, y))
    return model

def test_model(X, y, model):
    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print('--------------------------------')
    print('[RESULTS] Accuracy: %.2f' % accuracy)
    print('[RESULTS] Precision: %.2f' % precision)
    print('[RESULTS] Recall: %.2f' % recall)
    print('[RESULTS] F1: %.2f' % f1)
    print('--------------------------------')
    return pred


def main(image_dir_labelled, label_dir, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled,
         classifier, fr, output_model, x_train_save, y_train_save, x_test_save, y_test_save, unlabelled_save, save=True,
         use_saved=True):
    start = time.time()

    image_list_labelled, label_list_labelled = read_data_labelled(image_dir_labelled, label_dir, b_n_labelled,
                                                                  e_n_labelled)
    image_list_unlabelled = read_data_unlabelled(image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled)

    if use_saved:
        X_train = np.load(x_train_save)
        y_train = np.load(y_train_save)
        X_test = np.load(x_test_save)
        y_test = np.load(y_test_save)
        # X_unlabelled = np.load(unlabelled_save)
        print('Features loaded from file')
        print('[INFO] Feature vector size:', X_train.shape)

    else:
        X_train, X_test, y_train, y_test = create_training_dataset(image_list_labelled, label_list_labelled, test=True)
        # X_unlabelled = create_unlabelled_dataset(image_list_unlabelled)
        print('Feature extraction time in minutes:', (time.time() - start) / 60)

        if save:
            np.save(x_train_save, X_train)
            np.save(y_train_save, y_train)
            np.save(x_test_save, X_test)
            np.save(y_test_save, y_test)
            # np.save(unlabelled_save, X_unlabelled)

    model = train_model(X_train, y_train, classifier, fr, SSL=False)
    pred = test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))
    print('Total processing time in minutes:', (time.time() - start) / 60)
    print('Model saved as:', output_model)

    return pred

def SSL(threshold, image_dir_labelled, label_dir, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabeled,
        e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, unlabelled_save, save=True,
        use_saved=True):
    start = time.time()

    if use_saved:
        X_train = np.load(x_train_save)
        y_train = np.load(y_train_save)
        X_unlabelled = np.load(unlabelled_save)
        # X_unlabelled = X_train[:5]            #--- Deze uncommenten als je even snel wilt runnen als je features al hebt
        # X_train = X_train[5:10]
        # y_train = y_train[5:10]
        print('Features loaded from file')
        print('Feature size train {}'.format(X_train.shape))
        print('Feature size unlabelled {}'.format(X_unlabelled.shape))
    else:
        image_list_labelled, label_list_labelled = read_data_labelled(image_dir_labelled, label_dir, b_n_labelled,
                                                                      e_n_labelled)
        image_list_unlabelled = read_data_unlabelled(image_dir_unlabelled, b_n_unlabeled, e_n_unlabelled)
        X_train, y_train = create_training_dataset(image_list_labelled, label_list_labelled)
        X_unlabelled = create_unlabelled_dataset(image_list_unlabelled)
        print('Feature extraction time in minutes:', (time.time() - start) / 60)

        if save:
            np.save(x_train_save, X_train)
            np.save(y_train_save, y_train)
            np.save(unlabelled_save, X_unlabelled)

    ## Turn the X_train, y_train and X_unlabelled into a pandas dataframe something
    X_train = pd.DataFrame.from_dict(X_train)
    y_train = pd.DataFrame.from_dict(y_train)
    X_unlabelled = pd.DataFrame.from_dict(X_unlabelled)

    iterations = 0
    pseudo_labels = []

    # Assign value to initiate while loop
    high_prob = [1]

    total_added = 0

    while len(high_prob) > 0:
        model = train_model(X_train, y_train, classifier, fr)
        if (iterations == 11):
            break
        y_hat_train = model.predict(X_train)

        print(f"Iteration {iterations}")

        if len(X_unlabelled) > 0:
            print("Length of unlabelled: {}".format(len(X_unlabelled)))
            # Generate predictions for the unlabelled data
            print("Now predicting labels for unlabelled data")

            pred_probs = model.predict_proba(X_unlabelled)
            preds = model.predict(X_unlabelled)
            prob_0 = pred_probs[:, 0]
            prob_1 = pred_probs[:, 1]

            # Store predictions in pd dataframe
            df_pred_prob = pd.DataFrame([])
            df_pred_prob['preds'] = preds
            df_pred_prob['prob_0'] = prob_0
            df_pred_prob['prob_1'] = prob_1
            df_pred_prob.index = X_unlabelled.index

            # Separate predictions with > 99% probability
            high_prob = pd.concat(
                [df_pred_prob.loc[df_pred_prob['prob_0'] > threshold],
                 df_pred_prob.loc[df_pred_prob['prob_1'] > threshold]],
                axis=0)
            print(f"{len(high_prob)} high-probability predictions added to training data")

            total_added = total_added + len(high_prob)

            pseudo_labels.append(len(high_prob))

            # Really add them to x_train and the labels to y_train
            X_train = pd.concat([X_train, X_unlabelled.loc[high_prob.index]], axis=0)
            y_train = pd.concat([y_train, high_prob.preds])

            # Drop pseudo labeled instances from the unlabelled data.
            X_unlabelled = X_unlabelled.drop(index=high_prob.index)
            print(f"{len(X_unlabelled)} unlabelled instances remaining. \n")

        # update iterations counter
        iterations += 1

    print('Total unlabelled data added: {}'.format(total_added))
    pkl.dump(model, open(output_model, "wb"))
    print('Total processing time in minutes:', (time.time() - start) / 60)
    print('Model saved as:', output_model)

    return model

def create_features2(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, _ = create_features(img, img_gray, label=None, train=False, subsamplen=False)

    return features

def compute_prediction(img, model, save_f, use_saved_f, patient_slice, data_names, FEATURE_SAVE_PATH):
    image = np.mean(np.array(img), axis=2)
    image_shape = image.shape
    border = 5 # (haralick neighbourhood - 1) / 2

    img = cv2.copyMakeBorder(img, top=border, bottom=border, \
                                  left=border, right=border, \
                                  borderType = cv2.BORDER_CONSTANT, \
                                  value=[0, 0, 0])

    features_savefile = data_names + '_' + patient_slice + '.npy'
    features_save = os.path.join(FEATURE_SAVE_PATH, features_savefile)
    if use_saved_f:
        np.load(features_save)
    else:
        features = create_features2(img)
        if save_f:
            np.save(features_save, features)

    predictions = model.predict_proba(features.reshape(-1, features.shape[1]))
    inference_img = predictions[:,1].reshape(image_shape)
    mask_predicted1 = inference_img > 0.5
    mask_predicted2 = inference_img > 0.6
    mask_predicted3 = inference_img > 0.7
    mask_predicted4 = inference_img > 0.8
    mask_predicted5 = inference_img > 0.9

    return inference_img, predictions, features, img, mask_predicted1, mask_predicted2, mask_predicted3, mask_predicted4, mask_predicted5

def infer_images(image_dir, model, model_name, output_dir, data_names, FEATURE_SAVE_PATH, save_f = True, use_saved_f = False):

    filelist = glob(os.path.join(image_dir,'*.png'))

    print ('[INFO] Running inference on %s test images' %len(filelist))
    total = len(filelist)
    iteration = 1

    for file in filelist:
        print ('[INFO] Processing images:', os.path.basename(file))
        patient_slice = os.path.splitext(os.path.basename(file))[0]
        inference_img, predictions, features, img, mask_predicted1, mask_predicted2, mask_predicted3, mask_predicted4, mask_predicted5 = compute_prediction(cv2.imread(file, 1), model, save_f, use_saved_f, patient_slice, data_names, FEATURE_SAVE_PATH)
        infer_name = model_name + '_inf_' + patient_slice + '.png'
        image_name1 = model_name + '_mask1_' + patient_slice + '.png'
        image_name2 = model_name + '_mask2_' + patient_slice + '.png'
        image_name3 = model_name + '_mask3_' + patient_slice + '.png'
        image_name4 = model_name + '_mask4_' + patient_slice + '.png'
        image_name5 = model_name + '_mask5_' + patient_slice + '.png'

        cv2.imwrite(os.path.join(output_dir, infer_name), inference_img)
        cv2.imwrite(os.path.join(output_dir, image_name1), mask_predicted1 * 255)
        cv2.imwrite(os.path.join(output_dir, image_name2), mask_predicted2 * 255)
        cv2.imwrite(os.path.join(output_dir, image_name3), mask_predicted3 * 255)
        cv2.imwrite(os.path.join(output_dir, image_name4), mask_predicted4 * 255)
        cv2.imwrite(os.path.join(output_dir, image_name5), mask_predicted5 * 255)

        to_go = total - iteration
        print("Still {} more images to go".format(to_go))
        iteration = iteration + 1

#### MAIN CODE
model_name = 'features_SSL3'
data_names = 'features'
name = "Nathalie"

if name == "GPU":
    CLASS_DATA_PATH = '/home/8dm20-4/ClassData'
    MODEL_SAVE_PATH = '/home/8dm20-4/Models'
    FEATURE_SAVE_PATH = '/home/8dm20-4/Features'
    output_dir = '/home/8dm20-4/Output'
    image_dir_validation = os.path.join(CLASS_DATA_PATH, 'Validation/Slices')  # path to validation images
    image_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled/Slices')  # path to labelled images
    label_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled/Masks/Slices')  # path to labels
    image_dir_unlabelled = os.path.join(CLASS_DATA_PATH, 'Unlabelled/Slices')  # path to unlabelled data

if name == "Nathalie":
    CLASS_DATA_PATH = r'C:\Nathalie\Tue/Master\Jaar_1\Q3\Capita_Selecta\Project\Codes_def\ClassData'    #Data folder
    MODEL_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar_1\Q3\Capita_Selecta\Project\Models'                       #Define where the model should be saved
    FEATURE_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar_1\Q3\Capita_Selecta\Project\Features'
    output_dir = r'C:\Nathalie\Tue\Master\Jaar_1\Q3\Capita_Selecta\Project\Output'
    image_dir_validation = os.path.join(CLASS_DATA_PATH, 'Validation\Slices')  # path to validation images
    image_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled\Slices')  # path to labelled images
    label_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled\Masks\Slices')  # path to labels
    image_dir_unlabelled = os.path.join(CLASS_DATA_PATH, 'Unlabelled\Slices')  # path to unlabelled data

#paths
model_savefile = model_name +'.sav'
output_model = os.path.join(MODEL_SAVE_PATH, model_savefile)                                #specify path to save model
x_train_savefile = data_names + '_x_train.npy'
y_train_savefile = data_names + '_y_train.npy'
x_test_savefile = data_names + '_x_test.npy'
y_test_savefile = data_names + '_y_test.npy'
unlabelled_savefile = data_names + '_unlabelled.npy'

x_train_save = os.path.join(FEATURE_SAVE_PATH, x_train_savefile)
y_train_save = os.path.join(FEATURE_SAVE_PATH, y_train_savefile)
x_test_save = os.path.join(FEATURE_SAVE_PATH, x_test_savefile)
y_test_save = os.path.join(FEATURE_SAVE_PATH, y_test_savefile)
unlabelled_save = os.path.join(FEATURE_SAVE_PATH, unlabelled_savefile)

#Classifier properties
classifier = 'RF'                                          #options 'SVM', 'RF', 'GBC'
fr = True                                                  #specify feature reduction True/False (now PCA)

#Creating subset of data
b_n_labelled = 0                                                        #specify first labelled image
e_n_labelled = 5*86                                                     #specify last labelled image
b_n_unlabelled = 0                                                      #specify first unlabelled image
e_n_unlabelled = 7*86                                                   #specify last unlabelled image

threshold = 0.95                                                        #Threshold for adding the unlabelled images!

#code to train model, all needed functions are in train_utilities_new.py
# - save = als je de features bepaald wilt opslaan dit gebeurt onder de naam data_names, default = TRUE
# - use_saved = als je de features al een keer heb bepaald en deze wil gebruiken, hij gebruikt dan weer data_names, default=TRUE
#  LET OP ALS JE HEM NOG NOOIT HEB GERUND EN DE FEATURES DUS NOG NIET HEBT MOET JE USE_SAVED OP FALSE ZETTEN
# pred = main(image_dir_labelled, label_dir_labelled, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabelled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, x_test_save, y_test_save, unlabelled_save, save=True, use_saved=False)
model_pred_SSL = SSL(threshold, image_dir_labelled, label_dir_labelled, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabelled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, unlabelled_save, save = True, use_saved = True)


infer_images(image_dir_validation, model_pred_SSL, model_name, output_dir, data_names, FEATURE_SAVE_PATH, save_f=True, use_saved_f=False)