'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 1: Simple image search

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains code to do a k-nearest neighbors image search.
The script saves a csv file and a plot of the 5 most similar images in the 'out' folder.
'''

# loading packages
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# import from utils script
from search_utils import save_plot, extract_features

# tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16

# define argument parser and add arguments
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", help = "Name of folder where the images are located", default='flowers')
    parser.add_argument("--image", help = "Filename of target image", default='image_0001.jpg')
    parser.add_argument("--out_folder", help = "name of folder where output files are saved to", default='neighbor_search')
    args = vars(parser.parse_args())
    
    return args

# extract features
def features_from_folder(folder:str):
    '''
    Loops over each file in a directory and performs feature extraction on each image.

    Arguments:
    - folder: path to folder where images are located

    Returns:
    - filenames: names of files in the folder
    - feature_list: list of extracted features for each file
    '''

    # initialize VGG16 model, without the classification layers
    model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))

    # get all filenames in the folder
    filenames = [name for name in sorted(os.listdir(folder))]

    # initialize empty list
    feature_list = []

    # loop over filenames and use feature extractor function from utils script
    for i in range(len(filenames)):
        feature_list.append(extract_features(os.path.join(folder, filenames[i]), model))
    
    return filenames, feature_list

# find K-nearest neighbors
def find_neighbors(feature_list:list, filenames:list, target_image:str, out_folder:str):
    '''
    Calculate nearest neighbors to a target image using a feature list
    Saves a .csv file from a Pandas dataframe containing the 5 closest images

    Arguments:
    - feature_list: list of extracted features for each image
    - filenames: list of filenames for data directory
    - target_image: name of target image
    - out_folder: where to save the output csv

    Returns:
    - A Pandas series containing the names of the 5 nearest neighbors.
    '''

    # initialize K-nearest neighbors algorithm
    neighbors = NearestNeighbors(n_neighbors=10, 
                            algorithm='brute',
                            metric='cosine').fit(feature_list)

    # find the index of target image in filenames list
    file_idx = filenames.index(target_image)

    # save the indices and distances of the neighbors to the target image
    distances, indices = neighbors.kneighbors([feature_list[file_idx]])

    # initialize empty lists
    idxs = []
    dist = []
    
        # save the 5 closest images' indices and distances
    for i in range(1,6):
        idxs.append(indices[0][i])
        dist.append(distances[0][i])
    
    # save the filenames of the 5 closest images
    names = [filenames[i] for i in idxs]

    # create dataframe
    data = pd.DataFrame({"filename" : pd.Series(names),
                        "distance_score" : pd.Series(dist),
                        'index': pd.Series(idxs)})
    
    # save as csv
    data.to_csv(os.path.join("out", out_folder, f"{target_image}.csv"))
    
    # return filenames as a pandas series to be used in the plotting function
    return data['filename']

def neighbour_search(in_folder:str, target_image:str, out_folder:str):
    
    # define path to images
    path = os.path.join("data", in_folder)

    # save filenames and extract features for each image
    filenames, feature_list = features_from_folder(path)

    # find 5 nearest neighbors, save csv file and save filenames
    names = find_neighbors(feature_list, filenames, target_image, out_folder)

    # create and save plot of 5 closest images
    save_plot(path, names, target_image, out_folder)

def main():
    args = argument_parser()

    neighbour_search(args['in_folder'], args['image'], args['out_folder'])

if __name__ == '__main__':
    main()