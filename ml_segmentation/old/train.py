from train_utilities import *

#paths
image_dir = 'E:/CSMIA/2D_ML/im'                             #path to images
label_dir = 'E:/CSMIA/2D_ML/lb'                             #path to labels
output_model = 'E:/CSMIA/2D_ML/test_model.sav'              #specify path to save model

#Classifier properties
classifier = 'SVM'                                          #options 'SVM', 'RF', 'GBC'
fr = False                                                  #specify feature reduction True/False (now PCA)

#Creating subset of data
b_n = 0                                                     #specify first image
e_n = 5*86                                                  #specify last image 

#code to train model, all needed functions are in train_utilities.py
pred = main(image_dir, label_dir, b_n, e_n, classifier, fr, output_model)
