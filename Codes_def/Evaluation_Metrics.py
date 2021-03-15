from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import imageio
from sklearn.metrics import adjusted_rand_score
import seg_metrics.seg_metrics as sg
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
    TP = np.sum(true_pos)
    TN = np.sum(true_neg)
    FP = np.sum(false_pos)
    FN = np.sum(false_neg)
    VS = 1 - (abs(FN-FP))/(2*TP+FP+FN)
    AUC = 1 - ((1-sensitivity)+(1-specificity))/2
    return sensitivity, specificity, accuracy, VS, AUC

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


codes_def_path = 'E:/CSMIA/Codes_def/'
# test data
PATH_True = os.path.join(codes_def_path, 'TrainingData/p125/prostaat.mhd')
PATH_PREDICTED =os.path.join(codes_def_path, 'Results_atl_FINAL/transform_p125/Transformed_masks/mask_B_spline_p125/result.mhd')
TRUE_im = sitk.ReadImage(PATH_True)
TRUE_im = sitk.GetArrayFromImage(TRUE_im)[:,90:263,50:241]
PREDICTED_im = sitk.ReadImage(PATH_PREDICTED)
PREDICTED_im = sitk.GetArrayFromImage(PREDICTED_im)[:,90:263,50:241]

y_test = TRUE_im
y_pred = PREDICTED_im

sensitivity, specificity, accuracy, VS, AUC = calculate_sensitivity_specificity(y_test, y_pred)
dice = dice(y_pred, y_test)

y_test = y_test.ravel()
y_pred = y_pred.ravel()
ARS = adjusted_rand_score(y_test, y_pred)

print ('Sensitivity:', sensitivity)
print ('Specificity:', specificity)
print ('Accuracy:', accuracy)
print('Dice:', dice)


print('VS:',VS)
print('ARS', ARS)

