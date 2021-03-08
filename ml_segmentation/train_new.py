from train_utilities_new import *

name = "Nathalie"

if name == "GPU":
    CLASS_DATA_PATH = '/home/8dm20-4/ClassData'
    MODEL_SAVE_PATH = '/home/8dm20-4/Models'

if name == "Nathalie":
    CLASS_DATA_PATH = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project\Codes_def\ClassData'    #Data folder
    MODEL_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project'                       #Define where the model should be saved

model_name = 'test'

#paths
image_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled/Slices')                       #path to labelled images
label_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled/Masks/Slices')                          #path to labels
image_dir_unlabelled = os.path.join(CLASS_DATA_PATH, 'Unlabelled/Slices')                   #path to unlabelled data
model_savefile = model_name +'.sav'
output_model = os.path.join(MODEL_SAVE_PATH, model_savefile)                                #specify path to save model

#Classifier properties
classifier = 'SVM'                                          #options 'SVM', 'RF', 'GBC'
fr = False                                                  #specify feature reduction True/False (now PCA)

#Creating subset of data
b_n_labelled = 0                                                        #specify first labelled image
e_n_labelled = 5*86                                                     #specify last labelled image
b_n_unlabelled = 0                                                      #specify first unlabelled image
e_n_unlabelled = 7*86                                                   #specify last unlabelled image

#code to train model, all needed functions are in train_utilities_new.py
pred = main(image_dir_labelled, label_dir_labelled, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabelled, e_n_unlabelled, classifier, fr, output_model)
