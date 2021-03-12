from train_utilities_new import *

model_name = 'features_RF'
data_names = 'all_features'
name = "Nathalie"

if name == "GPU":
    CLASS_DATA_PATH = '/home/8dm20-4/ClassData'
    MODEL_SAVE_PATH = '/home/8dm20-4/Models'
    FEATURE_SAVE_PATH = '/home/8dm20-4/Features'

if name == "Nathalie":
    CLASS_DATA_PATH = r'C:\Nathalie\Tue/Master\Jaar_1\Q3\Capita_Selecta\Project\Codes_def\ClassData'    #Data folder
    MODEL_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar_1\Q3\Capita_Selecta\Project\Models'                       #Define where the model should be saved
    FEATURE_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar_1\Q3\Capita_Selecta\Project\Features'


#paths
image_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled\Slices')                       #path to labelled images
label_dir_labelled = os.path.join(CLASS_DATA_PATH, 'Labelled\Masks\Slices')                          #path to labels
image_dir_unlabelled = os.path.join(CLASS_DATA_PATH, 'Unlabelled\Slices')                   #path to unlabelled data
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
e_n_unlabelled = 1

threshold = 0.80                                                        #Threshold for adding the unlabelled images!

#code to train model, all needed functions are in train_utilities_new.py
# - save = als je de features bepaald wilt opslaan dit gebeurt onder de naam data_names, default = TRUE
# - use_saved = als je de features al een keer heb bepaald en deze wil gebruiken, hij gebruikt dan weer data_names, default=TRUE
#  LET OP ALS JE HEM NOG NOOIT HEB GERUND EN DE FEATURES DUS NOG NIET HEBT MOET JE USE_SAVED OP FALSE ZETTEN
pred = main(image_dir_labelled, label_dir_labelled, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabelled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, x_test_save, y_test_save, unlabelled_save, save=True, use_saved=False)
# pred_SSL = SSL(threshold, image_dir_labelled, label_dir_labelled, b_n_labelled, e_n_labelled, image_dir_unlabelled, b_n_unlabelled, e_n_unlabelled, classifier, fr, output_model, x_train_save, y_train_save, unlabelled_save, save = True, use_saved = True)