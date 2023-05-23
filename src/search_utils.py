'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 1: Simple image search

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains functions to plot similar image and to extract features from a single image for the k-nearest neighbor search.
'''
# import packages
import os
import sys
import cv2
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# tensorflow
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)


def save_plot(path, names, target_image, out_folder):
    '''
    Plot target image next to the five closest images

    Arguments:
    - path: path to folder where image data is stored
    - names: the filenames of the 5 closest images. Must be a pandas series
    - target_image: target input image
    - folder: specifies in what folder in output to save the plot to
    
    Returns:
    None
    '''
    
    # arrange plots
    f, axarr = plt.subplots(2, 3)
    
    # print target image
    axarr[0,0].imshow(mpimg.imread(os.path.join(path, target_image)))
    axarr[0,0].title.set_text('Target Image')

    # plot 5 most similar next to it
    axarr[0,1].imshow(mpimg.imread(os.path.join(path, names.iloc[0])))
    axarr[0,2].imshow(mpimg.imread(os.path.join(path, names.iloc[1])))
    axarr[1,0].imshow(mpimg.imread(os.path.join(path, names.iloc[2])))
    axarr[1,1].imshow(mpimg.imread(os.path.join(path, names.iloc[3])))
    axarr[1,2].imshow(mpimg.imread(os.path.join(path, names.iloc[4])))
    
    # remove axes from plot
    for ax in f.axes:
        ax.axison = False

    # save in output folder
    plt.savefig(os.path.join("out", out_folder, f"closest_to_{target_image}.png"))

def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)

    Arguments:
    - img_path: path to image to extract features on
    - model: model to use as a feature extractor

    Returns:
    - A list of image embeddings for the input image

    Source:
        This function is taken from the Session 10 Notebook for the Visual Analytics Course.
    """
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)

    return normalized_features