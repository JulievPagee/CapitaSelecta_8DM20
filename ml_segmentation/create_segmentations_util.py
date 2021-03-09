import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import pickle as pkl
import train_utilities_new
import math
import time

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory does not exist")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir" , help="Path to images", required=True)
    parser.add_argument("-m", "--model_path", help="Path to .p model", required=True)
    parser.add_argument("-o", "--output_dir", help="Path to output directory", required = True)
    args = parser.parse_args()
    return check_args(args)

def create_features(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, _ = train_utilities_new.create_features(img, img_gray, label=None, train=False)

    return features

def compute_prediction(img, model):
    image = np.mean(np.array(img), axis=2)
    image_shape = image.shape
    border = 5 # (haralick neighbourhood - 1) / 2

    img = cv2.copyMakeBorder(img, top=border, bottom=border, \
                                  left=border, right=border, \
                                  borderType = cv2.BORDER_CONSTANT, \
                                  value=[0, 0, 0])

    features = create_features(img)

    predictions = model.predict(features.reshape(-1, features.shape[1]))
    # pred_size = int(math.sqrt(features.shape[0]))
    inference_img = predictions.reshape(image_shape)

    return inference_img

def infer_images(image_dir, model_path, output_dir):

    filelist = glob(os.path.join(image_dir,'*.png'))

    print ('[INFO] Running inference on %s test images' %len(filelist))

    model = pkl.load(open( model_path, "rb" ) )

    for file in filelist:
        print ('[INFO] Processing images:', os.path.basename(file))
        inference_img = compute_prediction(cv2.imread(file, 1), model)
        patient_slice = os.path.splitext(os.path.basename(file))[0]
        image_name = 'mask_' + patient_slice + '.png'
        cv2.imwrite(os.path.join(output_dir, image_name), inference_img)

def main(image_dir, model_path, output_dir):
    start = time.time()
    infer_images(image_dir, model_path, output_dir)
    print("Total time to create masks:", (time.time()-start)/60)
