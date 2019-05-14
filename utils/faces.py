# faces.py ---
#
# Filename: faces.py
# Maintainer: Anmol Mann
# Description:
# Course Instructor: Kwang Moo Yi

# Code:

import os, sys

import numpy as np
import pickle
from utils.external import unpickle


def load_data(data_dir, data_type):
    """Function to load data from FACES.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted FACES files.

    data_type : string
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------
    data : ndarray (uint8)
        Data from the FACES dataset corresponding to the train/test
        split. The datata should be in NHWC format.

    labels : ndarray (int)
        Labels for each data. Integers ranging between 0 and 9.

    """

    file_dir = os.path.join(data_dir, "imgs")
    csv_file = ""
    if data_type == "train":
        csv_file = os.path.join(data_dir, "train_ids.csv")
    elif data_type == "test":
        csv_file = os.path.join(data_dir, "test_ids.csv")
    else:
        raise ValueError("Wrong data type {}".format(data_type))
        sys.exit()

    """
    # Load data from a csv file.
    temp_csv_ids = np.loadtxt(csv_file, dtype = np.str, delimiter = '\t')
    csv_ids = []
    # fetch everything from 0 till last 5th element
    for index in temp_csv_ids:
        csv_ids.append(index[:-4])
    """
    # Load data from Action Units File
    AU_file = os.path.join(data_dir, "aus_openface.pkl")
    AU_data = "" #  Nothing in here yet
    with open(AU_file, 'rb') as file_content:
        AU_data = pickle.load(file_content, encoding = "latin1")
    """
    # fetch common ids from csv file and keys of AUs (from its key, value pair)
    large_set_ids = set(csv_ids)
    large_set_AU = set(AU_data.keys())
    common_ids = large_set_ids.intersection(large_set_AU)
    common_ids = list(common_ids)
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        imgs_path = [os.path.join(file_dir, line.strip()) for line in lines]
        imgs_path = sorted(imgs_path[0:140000])

    #return common_ids, AU_data
    return imgs_path, AU_data


#
# faces.py ends here
