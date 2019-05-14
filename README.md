## ReImplementation of GANimation: Anatomically-aware Facial Animation from a Single Image

Its a GAN architecture conditioned on Action Units (AU) annotations. This model generates facial expressions in a continuous domain. This work is based on this [paper](https://arxiv.org/abs/1807.09251).
This approach can control the intensity of the desired expression.

<div align = "center">
<img src="imgs_gif/1.gif">
<img src="imgs_gif/2.gif">
<img src="imgs_gif/3.gif">
</div>

### Prerequisites

Install the packages in the requirement.txt file.

### Dataset Used

EmotioNet and celebA dataset.

The data directory is similar to the actual [project implementation](https://github.com/albertpumarola/GANimation/).

[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) toolkit is used to produce extract AUs from the images as well aligning them to size 128x128.

    - *FeatureExtraction.exe -fdir imgs/ -aus -simalign -nobadaligned -au_static -simsize 128 -format_aligned jpg -nomask*
    - I used Openface (.exe) on Windows platform.

*The celebA data's aus_openface.pkl, train_ids.csv, and test_ids.csv files can be downloaded at [this link](https://drive.google.com/file/d/1qxvOmTpukbNHJsY5MyEq05SlvvtMpZp6/view).*
    
### Training

The model is trained on Compute Canada. So, a bunch of different script files were used to run the jobs.

    [1] celebA_job.sh - Review this script to train the model on celebA dataset.
    [2] EmotioNet_job.sh - Review this script to train the model on EmotioNet dataset.
    
    *model.py* - has the generator and discriminator model architecture.
    *solution.py* - has both the training and testing functionalities of the model.
    *utils/* - has all the code used for loading the data into the model.
    
Model Checkpoint, logs and some test results are provided at **[HERE](https://drive.google.com/open?id=11tLBd2SfgGXkchKyUzg69WLkqlqgaDj7)**.
    

### Data Preparation
Scripts and python code in the EmotioNet directory is used to extract images and their Action Units from the data.
These scripts and codes have not been linked to the main implementation directly. But they were used in the first place to generate the EmotioNet dataset in order to provide the input data to the model to be trained upon.

When unzipping the celebA and EmotioNet dir. Create imgs/ directory in both of them. This img dir will consists of all the images included in the dictionary in aus_openface.pkl.

### Testing

    [1] celebA dataset - python3 solution.py --mode test
    [2] EmotioNet dataset - python3 solution.py --mode test --data_dir EmotioNet

    To generate gif images, use flag --as_gif True during testing.

Some of the results included losses and interpolations (GIF and JPG images generated by me during re-implementation) are provided in the *results.zip* folder.

##### Licenses

Please review the LICENSE file to review all the code references used for the re-implementation.

``NOTE: The base code is referenced from the Assignments completed during the course CSC 586B.``

##### Author

Anmol Mann, anmolmann@uvic.ca