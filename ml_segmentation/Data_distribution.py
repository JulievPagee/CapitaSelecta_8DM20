import numpy as np
import os
import SimpleITK as sitk
import imageio
import shutil

##Data distribution
DATA_PATH = 'C:/Nathalie/Tue/Master/Jaar 1/Q3/Capita Selecta/Project/Codes_def/TrainingData'
SAVE = 'ClassData'
SAVE_LABEL = 'ClassData/Labelled'
SAVE_MASKS = 'ClassData/Labelled/Masks'
SAVE_UNLABELED = 'ClassData/Unlabelled'
SAVE_VAL = 'ClassData/Validation'

if os.path.exists(SAVE) is False:
    os.mkdir(SAVE)
if os.path.exists(SAVE_LABEL) is False:
    os.mkdir(SAVE_LABEL)
if os.path.exists(SAVE_MASKS) is False:
    os.mkdir(SAVE_MASKS)
if os.path.exists(SAVE_UNLABELED) is False:
    os.mkdir(SAVE_UNLABELED)
if os.path.exists(SAVE_VAL) is False:
    os.mkdir(SAVE_VAL)

patient_list = os.listdir(DATA_PATH)
Val = ['p133', 'p119', 'p125']
Unlabeled = ['p102', 'p109', 'p115', 'p117', 'p120', 'p128', 'p129']
Label = ['p107', 'p116', 'p135', 'p127', 'p108']

for j in range(15):
    # select the data category of each image
    if patient_list[j] in Val:
        category = 'Validation'
    elif patient_list[j] in Label:
        category = 'Labelled'
    elif patient_list[j] in Unlabeled:
        category = 'Unlabelled'
    SAVE_FOLD = 'ClassData/{}/'.format(category)

        # load image
    img_path = os.path.join(DATA_PATH, patient_list[j], 'mr_bffe.mhd')
    image = imageio.imread(img_path)
    # save images
    save_path = SAVE_FOLD + 'img_' + patient_list[j] + '.mhd'
    image = sitk.GetImageFromArray(image)
    image = sitk.WriteImage(image, save_path)

    if category == 'Labelled':
        SAVE_FOLD_Masks = os.path.join(SAVE_FOLD, 'Masks/')
        img2_path = os.path.join(DATA_PATH, patient_list[j], 'prostaat.mhd')
        save_path2 = SAVE_FOLD_Masks + 'mask_' + patient_list[j] +'.mhd'
        img2_path_raw = os.path.join(DATA_PATH, patient_list[j], 'prostaat.zraw')
        save_path2_raw = SAVE_FOLD_Masks + 'mask_' + patient_list[j] + '.raw'

        shutil.copyfile(img2_path, save_path2)
        shutil.copyfile(img2_path_raw, save_path2_raw)
        print('mask patient '+patient_list[j]+'saved')

print('Images are devided into labelled, unlabelled and validation')