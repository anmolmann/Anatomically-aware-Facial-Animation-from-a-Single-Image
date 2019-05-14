# datawrapper.py ---
#
# Filename: datawrapper.py
# Maintainer: Anmol Mann
# Description:
# Course Instructor: Kwang Moo Yi

# Code:

import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image

from utils.faces import load_data


class FacesDataset(data.Dataset):
    """The dataset wrapper for FACES.

    While writing this class is completely unecessary if we were to obtain the
    FACES data in raw format since PyTorch already has it written. However,
    in our case we will also try to extract data, we will implement it
    ourselves.

    """

    def __init__(self, config, mode):
        """Initialization.
        
        Parameters
        ----------

        config:
            Configuration object that holds the command line arguments.
        
        mode: str
            A string detailing which split for this dataset object to load. For
            example, train, test.
        

        Notes
        -----

        By default, we assume the dataset is at downloads

        """

        # Save configuration
        self.config = config

        print(
            "Loading Dataset from {} for {}ing ...".format(config.data_dir, mode), end=""
        )

        """
        # Load data (note that we now simply load the raw data.
        common_ids, AU_data = load_data(config.data_dir, mode)
        # ids of the images
        self.data = common_ids
        # dictionary with img_ids as keys, and AUs as values
        self.label = AU_data
        """
        self.data, self.label = load_data(config.data_dir, mode)

        list_trans = []
        # Data augmentation and normalization for training
        if mode == "train":
            list_trans.append(transforms.RandomHorizontalFlip())

        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        if config.resize == True:
            # list_trans.append(transforms.Resize([138, 138], Image.BICUBIC))  # resize to 128 x 128
            # now, crop img
            list_trans.append(transforms.RandomCrop([128, 128]))
        else:
            list_trans.append(transforms.Resize(128))  # resize to 128 x 128
        list_trans.append(transforms.ToTensor()) #to convert the numpy images to torch images
        list_trans.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.list_trans = transforms.Compose(list_trans)

        print(" done.")

    def __len__(self):
        """ Returns number of samples. """

        # return the number of elements at `self.data`
        return len(self.data)

    def __getitem__(self, index):
        """Function to grab one data sample 
        
        Parameters
        ----------
        
        index: int
            Index to the sample that we are trying to extract.

        
        Returns
        -------

        data_cur: torch.Tensor
            one image
        
        label_cur: int
            corresponding attribute label
        """

        # load original image
        data_cur = Image.open(self.data[index]).convert("RGB")
        # make pytorch object and normalize it
        data_cur = self.list_trans(data_cur)

        # Also, normalize Aus between 0 and 1.
        img_id = str(os.path.splitext(os.path.basename(self.data[index]))[0])
        label_cur = torch.FloatTensor(self.label[img_id]/5.0)

        # print(img_name, label_cur)
        return data_cur, label_cur


#
# datawrapper.py ends here
