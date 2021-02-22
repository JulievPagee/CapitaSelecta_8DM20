from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import imageio
#import tensorflow as tf

def calculate_sensitivity_specificity(y_test, y_pred):
    # Note: More parameters are defined than necessary.
    # This would allow return of other measures other than sensitivity and specificity
    # NOTE: reference: https://pythonhealthcare.org/tag/specificity/

    # Get true/false for whether a breach actually occurred
    actual_pos = y_test == 1
    actual_neg = y_test == 0

    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (y_pred == 1) & (actual_pos)
    false_pos = (y_pred == 1) & (actual_neg)
    true_neg = (y_pred == 0) & (actual_neg)
    false_neg = (y_pred == 0) & (actual_pos)

    # Calculate sensitivity and specificity
    sum_true_pos = np.sum(true_pos)
    sum_false_neg = np.sum(false_neg)
    sensitivity = np.sum(true_pos) / (sum_false_neg+ sum_true_pos)
    specificity = np.sum(true_neg) / (np.sum(true_neg)+np.sum(false_pos))
    print('TP=',np.sum(true_pos), ',TN =',np.sum(true_neg), ',FP=',np.sum(false_pos), ',FN=', np.sum(false_neg))

    # Calculate accuracy
    accuracy = (np.sum(true_pos) + np.sum(true_neg)) / (
                np.sum(true_neg) + np.sum(true_pos) + np.sum(false_neg) + np.sum(false_pos))

    return sensitivity, specificity, accuracy

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

# test data
PATH_True = os.path.join(r'C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\TrainingData\TrainingData\p102\prostaat.mhd')
PATH_PREDICTED =os.path.join(r'C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\TrainingData\TrainingData\p102\prostaat.mhd')
TRUE_im = sitk.ReadImage(PATH_True)[:,:,20]
TRUE_im = sitk.GetArrayFromImage(TRUE_im)
PREDICTED_im = sitk.ReadImage(PATH_PREDICTED)[:,:,20]
PREDICTED_im = sitk.GetArrayFromImage(PREDICTED_im)

#fig, ax = plt.subplots(1, 2, figsize=(20, 5))
#ax[0].imshow(TRUE_im, cmap='gray')
#ax[0].set_title('True')
#ax[1].imshow(PREDICTED_im, cmap='gray')
#ax[1].set_title('Predicted')
#plt.show()

y_test = TRUE_im
y_pred = PREDICTED_im

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(y_test, y_pred)
dice = dice(y_pred, y_test)
print ('Sensitivity:', sensitivity)
print ('Specificity:', specificity)
print ('Accuracy:', accuracy)
print('Dice:', dice)
