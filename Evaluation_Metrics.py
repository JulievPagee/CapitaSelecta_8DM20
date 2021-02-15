from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
import imageio


def calculate_sensitivity_specificity(y_test, y_pred_test):
    # Note: More parameters are defined than necessary.
    # This would allow return of other measures other than sensitivity and specificity
    # NOTE: reference: https://pythonhealthcare.org/tag/specificity/

    # Get true/false for whether a breach actually occurred
    actual_pos = y_test == 1
    actual_neg = y_test == 0

    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (y_pred_test == 1) & (actual_pos)
    false_pos = (y_pred_test == 1) & (actual_neg)
    true_neg = (y_pred_test == 0) & (actual_neg)
    false_neg = (y_pred_test == 0) & (actual_pos)

    # Calculate accuracy
    accuracy = (np.sum(true_pos)+np.sum(true_neg))/(np.sum(true_neg)+np.sum(true_pos)+np.sum(false_neg)+np.sum(false_pos))

    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / (np.sum(true_pos)+np.sum(false_neg))
    specificity = np.sum(true_neg) / (np.sum(true_neg)+np.sum(false_pos))

    return sensitivity, specificity, accuracy


def dice_coef(y_true, y_pred, smooth=1):
    #NOTE: reference with additional info: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    # https://towardsdatascience.com/image-segmentation-choosing-the-correct-metric-aa21fd5751af
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    f1 = (2 * intersection + smooth) / (denominator + smooth)

    return (1 - f1) * smooth

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(y_test, y_pred_test)
dice = dice_coef(y_true, y_pred, smooth=1)
print ('Sensitivity:', sensitivity)
print ('Specificity:', specificity)
print ('Accuracy:', accuracy)
print('Dice:', dice)
