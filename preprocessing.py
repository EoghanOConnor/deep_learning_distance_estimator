# The KITTI dataset labels file is loaded.
# Once loaded all info about each object within the training dataset and append to all_label_info
# The info regarding the pedestrians is saved locally.

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import pickle
import os

def get_all_data(file):
    """
    This function gets data from the inputted file

    Parameters:
    file (.txt): 

    Returns:
    list of int, floats, strings: The infomation of the objects in the image
    """
    labels = []
    with open(file) as f:
        content = f.readlines()
    for obj in content:
        labels.append(obj.split(' '))
    return labels

def get_inputs(results):
    """
    Extracts bounding box coordinates from the results object.

    This function iterates over the results object, checks if the last element of 
    each item is 0 (pedestrian class), and if so, appends the first four elements (which are expected to 
    be bounding box coordinates) to the inputs list.

    If any exception occurs during this process, the function prints 'Empty' and
    does not return a value.

    Args:
        results (object): The output object from a model's prediction method.

    Returns:
        list: A list of bounding box coordinates.

    Exceptions:
        Any exceptions that occur during processing are caught and result in 'Empty' being printed.
    """
    try:
        inputs = []
        out = results.xyxy[0].numpy()
        for obj in out:
            if obj[-1] == 0:
                inputs.append(obj[0:4])
                
        return inputs
    
    except:
        print('Empty')


def main():
    # Directory to KITTI dataset label information
    train_label_dir = 'data_object_label_2/training/label_2/'

    # Retrieve all info about each object within the training dataset and  append to all_label_info.
    all_label_info = []
    for filename in os.listdir(train_label_dir):
        label_file = train_label_dir + filename
        info = get_all_data(label_file)
        for obj in info:
            all_label_info.append(obj)

    # label the columns of the info
    columns = ['class Names', 'truncatation', 'occlusion','alpha','xmin','ymin','xmax','ymax','height','width','length','xloc','yloc','zloc','rotation_y']
    df = pd.DataFrame (all_label_info, columns = columns)

    #retrieve only info about pedestrians and save it locally
    pedestrian_info = df.loc[df['class Names'] == 'Pedestrian']

    pedestrian_info.to_csv('pedestrians.csv')

    
if __name__ == "__main__":
    main()