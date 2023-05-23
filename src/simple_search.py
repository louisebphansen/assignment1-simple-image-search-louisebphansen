'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 1: Simple image search

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains code to do a simple image search based on color histograms.
The script saves a csv file and a plot of the 5 most similar images in the 'out' folder.
'''

# loading needed packages
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# import functions from utils script
from search_utils import save_plot

# add argument parser and arguments
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", help = "Name of folder where the images are located", default='flowers')
    parser.add_argument("--image", help = "Filename of target image", default='image_0001.jpg')
    parser.add_argument("--out_folder", help = "name of folder where output files are saved to", default='simple_search')
    args = vars(parser.parse_args())
    
    return args
    
# function to calculate and normalize a color histogram
def hist_fun(filepath:str, filename:str):
    
    '''
    This function loads an image file, calculates a color histogram and normalizes the histogram.

    Arguments:
    - filename: the image to create and normalize a color histogram from
    - filepath: the filepath to the folder where the image is

    Returns:
    - Normalized color histogram

    '''
    # loading the input image
    im = cv2.imread(os.path.join(filepath, filename)) 
    
    # calculate histogram for all three channels, using no mask, 8 pixel bins and 0-255 range
    hist = cv2.calcHist([im], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256]) 

    # normalizing the histogram using min-max normalization
    hist_norm = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

    return hist_norm

# function that compares color histograms and saves the closest images
def compare_distance(filepath:str,filename:str, out_folder:str):

    '''
    This function calculates a color histogram from an input image and compares it to every other image in the input filepath by distance scores. 
    The function saves a Pandas DataFrame containing the 5 closest images measured by distance scores and saves it as a csv file in the "out" folder.

    Arguments:
    - filename: the filename of the chosen input image. The file must be located in the "filepath" directory.
    - filepath: the filepath to the folder where the chosen input image and the images to be compared are located.
    - out_folder: name of folder where csv file is saved

    Returns:
    - The names of the five closest images by distance scores
    '''

    # using my "hist_fun" function to read the target image file, calculate the color histogram and normalize it
    hist1_norm = hist_fun(filepath, filename)
    
    # creating two empty lists to append distance scores and filenames to 
    distances = []
    filenames = []

    # loop over every file in the input filepath
    for other_file in os.listdir(filepath):
        if other_file == filename: # if the file is the target filename, continue to next iteration
            continue
        else:
            hist2_norm = hist_fun(filepath, other_file) # calculate and normalize color histogram for file

            # compare the histograms of the target image and the new image by calculating distance scores (rounded to two decimals)
            distance_score = round(cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_CHISQR), 2)

            # append the filenames and scores to the "distances" and "filenames" lists
            distances.append(distance_score)
            filenames.append(other_file)

    # create dataframe from the two lists by converting them to pandas-series
    data = pd.DataFrame({"filename" : pd.Series(filenames),
                        "distance_score" : pd.Series(distances)})

    # sorting the dataframe by the distance scores, in ascending order
    sorted_data = data.sort_values('distance_score', ascending = True) 

    # extracting first five rows, i.e., the five closest images by distance scores
    closest_images = sorted_data[0:5]

    # saving the 5 closest images to a csv file in the "out" folder
    closest_images.to_csv(os.path.join("out", out_folder, f"{filename}.csv"))

    closest_names = closest_images['filename']

    # return the names of the 5 closest images
    return closest_names

# compare color histograms and save closest images
def simple_search(in_folder:str, target_image:str, out_folder:str):

    # define path to images
    path = os.path.join("data", in_folder)

    # compare distances and save 5 closest to a csv file
    five_closest = compare_distance(path, target_image, out_folder)

    # create and save plot of target image next to the five closest images using plotting function from utils script
    save_plot(path, five_closest, target_image, out_folder)

def main():
    args = argument_parser()

    simple_search(args['in_folder'], args['image'], args['out_folder'])

if __name__ == '__main__':
    main()







