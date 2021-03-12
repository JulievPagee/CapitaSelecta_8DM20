import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image

if name == "Anouk":
    CLASS_DEF_PATH = r'C:\Users\20165272\Documents\8DM20 Capita Selecta\Project\ml_segmentation'    #Data folder

CODES_DEF_PATH = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project\Codes_def'

Labelled = os.path.join(CODES_DEF_PATH, 'ClassData', 'Labelled')
Labels = os.path.join(CODES_DEF_PATH, 'ClassData', 'Labelled', 'Masks')
Unlabelled = os.path.join(CODES_DEF_PATH, 'ClassData', 'Unlabelled')
Validation = os.path.join(CODES_DEF_PATH, 'ClassData', 'Validation')

Label_list = os.listdir(Labelled)
Labels_list = os.listdir(Labels)
Unlabelled_list = os.listdir(Unlabelled)
Validation_list = os.listdir(Validation)

#Make lists for folders for slices
Labelled_slice = os.path.join(Labelled, 'Slices')
Labels_slice = os.path.join(Labels, 'Slices')
Unlabelled_slice = os.path.join(Unlabelled, 'Slices')
Validation_slice = os.path.join(Validation, 'Slices')

if os.path.exists(Labelled_slice) is False:
    os.mkdir(Labelled_slice)
if os.path.exists(Labels_slice) is False:
    os.mkdir(Labels_slice)
if os.path.exists(Unlabelled_slice) is False:
    os.mkdir(Unlabelled_slice)
if os.path.exists(Validation_slice) is False:
    os.mkdir(Validation_slice)

# IMG_Label = []
# IMG_Labels = []
# IMG_Unlabelled = []

nr_slices = 86

for i in Label_list:
    if ".mhd" in i:
        path = os.path.join(Labelled, i)
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        patientnr = i[4:8]
        for slice in range(0,nr_slices):
            save_name = 'img_' + patientnr + '_slice_' + str(slice) + '.png'
            save_path = os.path.join(Labelled_slice, save_name)
            image = img[slice,:,:]
            image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
            image_slice = Image.fromarray(image)
            # stacked_image = np.stack((image,) * 1, axis=-1)
            # IMG_Label.append(stacked_image)
            # image = sitk.WriteImage(image, save_path)
            image_slice.save(save_path)
        print(patientnr + ' done')

# print('Labeled sizes: ' + str(len(IMG_Label)))

for k in Label_list:
    if '.mhd' in k:
        patientnr = k[4:8]
        path = os.path.join(CODES_DEF_PATH, 'TrainingData', patientnr, 'prostaat.mhd')
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        for slice in range(0,nr_slices):
            save_name = 'mask_' + patientnr + '_slice_' + str(slice) + '.png'
            save_path = os.path.join(Labels_slice, save_name)
            image = img[slice,:,:]
            image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
            image_slice = Image.fromarray(image)
            # stacked_mask = np.stack((image,) * 1, axis=-1)
            # IMG_Labels.append(stacked_mask)
            image_slice.save(save_path)
        print(patientnr + ' done')

# print('Labels: ' + str(len(IMG_Labels)))

for l in Unlabelled_list:
    if '.mhd' in l:
        path = os.path.join(Unlabelled,l)
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        patientnr = l[4:8]
        for slice in range(0,nr_slices):
            save_name = 'img_' + patientnr + '_slice_' + str(slice) + '.png'
            save_path = os.path.join(Unlabelled_slice, save_name)
            image = img[slice,:,:]
            image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
            image_slice = Image.fromarray(image)
            # stacked_image = np.stack((image,) * 1, axis=-1)
            # IMG_Unlabelled.append(stacked_image)
            image_slice.save(save_path)
        print(patientnr + ' done')

for m in Validation_list:
    if '.mhd' in m:
        path = os.path.join(Validation,m)
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        patientnr = m[4:8]
        for slice in range(0,nr_slices):
            save_name = 'img_' + patientnr + '_slice_' + str(slice) + '.png'
            save_path = os.path.join(Validation_slice, save_name)
            image = img[slice,:,:]
            image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
            image_slice = Image.fromarray(image)
            # stacked_image = np.stack((image,) * 1, axis=-1)
            # IMG_Unlabelled.append(stacked_image)
            image_slice.save(save_path)
        print(patientnr + ' done')

# print('Unlabelled: ' + str(len(IMG_Unlabelled)))

# np.save(os.path.join(CODES_DEF_PATH,'ClassData','Labelled_array'),np.array(IMG_Label))
# np.save(os.path.join(CODES_DEF_PATH,'ClassData','Labels_array'),np.array(IMG_Labels))
# np.save(os.path.join(CODES_DEF_PATH,'ClassData','Unlabelled_array'),np.array(IMG_Unlabelled))
