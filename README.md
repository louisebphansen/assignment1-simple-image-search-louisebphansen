
# Assignment 1: Building a simple image search algorithm
This assignment is the first assignment for the Visual Analytics course on the elective in Cultural Data Science at Aarhus University, 2023. 

### Contributions
The code was created by me, but code provided in the notebooks throughout the course have been reused.

### Assignment description

- Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric

### Contents of the repository

| Folder/File  | Contents| Description |
| :---:   | :---: | :--- |
|```data```|flowers.zip| The folder contains the zipped data for this assignment. The data consists of 1360 images of different flowers.|
|```out``` | ```neighbor_search``` ```simple_search```| The folder contains the output from running either the neighbour search or the simple search scripts. The subfolders contain a plot and a csv file showing the five closest images |
|```src```|**neighbor_search.py**, **simple_search.py**, **utils.py**| The folder contains Python scripts to run a K-nearest neighbor image search and a simple image search using color histograms. **utils.py** contains various functions for the two scripts. |
|README.md|-|Description of the repository and how to run the code.|
|requirements.txt|-|Packages required to run the code.|
|run.sh|-|Bash script for running both search-scripts with default arguments.|
|setup.sh|-|Bash script for creating virtual environment for the assignment.|


### Methods
*The following section describes the methods used in the provided Python scripts*

#### Simple Image Search
```simple_search.py```contains code to perform a simple image search by comparing color histograms, a basic method of comparing differences and similarities of pixel values across color channels. The script calculates and normalizes a color histogram for a chosen target image. Next, it calculates a histogram for every *other* image in a folder and calculates the distance score, based on a chi-squared metric, between the target image's histogram and each of all the other images' histograms. A csv file and a plot with the five closest images based on distance scores are saved in the ```simple_search``` folder in the ```out``` folder.

#### k-Nearest Neigbor Image Search
```neighbor_search.py``` contains code to perform image search based on a k-Nearest Neighbor (kNN) algorithm and searches for similar images to a target image based on dense image embeddings. A pretrained convolutional neural network (CNN), *VGG16*, is used without its classification layers to extract features from each image in the folder, which results in a list with image embeddings for *each* image, i.e., a dense, 1D representation of each image's features. This is used in a kNN algorithm to find the most similar images, in this case, by calculating the cosine distances between image embeddings.  Similarly to the simple-search script, a csv file and a plot of the 5 most similar images are saved in the ```neighbor_search``` folder in the ```out``` folder. 

### Usage

All code for this assignment was designed to run on an *Ubuntu 22.10* operating system.

To reproduce the results in this repository, clone it using ```git clone```.

It is important that you run all scripts from the *assignment1-simple-image-search-louisebphansen* folder, i.e., your terminal should look like this:
```
--your_path-- % assignment1-simple-image-search-louisebphansen %
```

To get the data, unzip the *flowers.zip* file, and place the unzipped folder, ```flowers```, inside the ```data``` folder.

#### Setup
First, ensure that you have installed the *venv* package for Python (if not, run ```sudo apt-get update``` and ```sudo apt-get install python3-venv```).

To setup the virtual environment, run ```bash setup.sh``` from the terminal (again, still from the main folder).

#### Run code
To run the code, you can do the following:

##### Run script(s) with predefined arguments

From the terminal, type ```bash run.sh```, which activates the virtual environment and runs both the **simple_search.py** script and the **neighbor_search.py** script with default arguments, i.e., uses the ```flowers``` folder as data, uses the first image, *image_0001.jpg* as the target image and saves the output in either the ```simple_search``` or the ```neighbor_search``` folders. 

##### Define arguments yourself
Alternatively, you can run the scripts separately or define the arguments yourself. From the terminal, activate the virtual environment and run the script(s) with the desired arguments:

```
source env/bin/activate # activate the virtual environment

python3 src/simple_search.py --in_folder <input_folder> --image <target_image> --out_folder <output_folder>

# and/or

python3 src/neighbor_search.py --in_folder <input_folder> --image <target_image> --out_folder <output_folder>
```
**Arguments:**

- **input_folder:** Name of a folder in the ```data``` folder containing the images. Default: 'flowers'
- **target_image:** The full name (i.e., including filetype ending such as .jpg) of the image we want to find the most similar to. Default: 'image_0001.jpg'
-  **output_folder:** Name of the folder in ```out``` where the results should be saved. Default for *simple_search.py* is 'simple_search', default for *neighbor_search.py* is 'neigbor_search'.


### Results

Results from running the scripts using 'image_0001.jpg' as the target image is saved in the ```out``` folder. The following pictures show the five closest images to the target image for the two methods:

#### Simple search algorithm
![image](https://user-images.githubusercontent.com/75262659/236681885-bcce0e64-44d2-41ba-b2fe-49ba1d93ac06.png)

#### k-Nearest Neighbor image search
![image](https://user-images.githubusercontent.com/75262659/236681935-f2d2eb73-d055-4424-9eef-25ef3b28cf27.png)

When running the simple search algorithm, which compares color histograms, the 5 closest images are somewhat similar in color schemes to the target image. Except for the image of the white flower, they contain similar yellow, brown and green colors. Although many of the colors match the target image's, none of the images contain the *same* flower as the one in the target image. As the algorithm is only comparing distributions of pixel color values, it makes sense that it is fairly good at finding images with similar ranges of color values, but *not* similar image content. 

Looking at the output from the kNN image algorithm, it is clear that this is a much better way of performing image search. It appears to be better at finding both images with similar colors *and* similar content, as all the images are of the same flower as the target image and the same color. It also appears as if the composition of the images are similar; all images are closeups of the yellow flower, where the image is taken a bit from underneath. They all also contain green grass at the bottom of the images. As this algorithm uses the VGG16 model as a feature extractor, it has most likely been able to extract more sophisticated features from the image than the simple color-histogram search algorithm, such as shapes and perspectives, which provides for a much better way of doing image search. 


