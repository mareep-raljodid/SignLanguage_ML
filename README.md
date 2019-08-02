#                 [![modeled with tensorflow](https://drive.google.com/uc?authuser=0&id=1Vfwy9Cb3KF_ATWPIeifcP_L286EUf0_3&export=download)](https://www.tensorflow.org/) Sign_Language_ML

#                  

[![made with python](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org)



### A Machine Learning Model trained on American sign language Dataset to predict the letters or numbers from a picture of a hand doing a certain sign.

Want a Fast Track look and see if/how this works?

Have a look here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMvokUcPU1LJ8SvsWKLiEk9whFvVUWi3)




- Datasets provided by Kaggle [unpublished]
- Model based on using Convulations and Prediction using RELU activation on Gradient Descent.
- Sparse Categorical Crossentropy
- Accuracy Score with validation set: 95%

#### Dependencies: 
- TensorFlow
- Random
- Matplotlib
- OpenCV
- OS
- Numpy

    + Make sure you have pip3 installed, you can do that by (for linux):
    ```
    sudo apt install python3-pip
    ```

    + And this (for Mac OSX), in the python3 directory:
    ```
    python3 get-pip.py
    ```

    + Install dependencies using the commands below (pip3 and python3 are required):

    ```bash
    pip3 install tf-nightly
    pip3 install random
    pip3 install matplotlib
    pip3 install opencv-python
    pip3 install os
    pip3 install numpy
    ```

#### Information about the files present here:

- env_name: Required Packages for runtime and dependencies.

- Pretrained_Model: Contains the checkpoints and epocs saved model which took a total of 13 hours to train. 
    + You're Welcome :)

- photo/picture [jpg]: photo.jpg is R and picture.jpg is D, these are test images and examples of how an ideal image    that can be accurately predicted by this model looks.

- SL_fullNB.ipynb: The full python notebook containing the script snippets to train and predict.

- SL_independent.py: Full python script to self train and predict images.

- SL_quickPredict.py: Predicts an image based on the saved model, no training required, 13 hours saved. 



#### Instructions:

  + For a Quick Prediction of a picture you have, compile the SL_quickPredict.py using the command below:
     
     ```bash
     python3 L_quickPredict.py
     ```
    and insert the path to your picture with the extention of the picture.

  + For a full re-train (CAUTION MAY TAKE 13 TO 16 HOURS!) just compile the SL_independent.py file with the training    data in same directory as the python script.
    You can get the training data from this [Google Drive link](https://drive.google.com/open?id=1apmXyY8OQx68b4-2G9Mttgrml7bldWfq)


![GitHub contributors](https://img.shields.io/github/contributors/mareep-raljodid/SignLanguage_ML?style=for-the-badge)

##### Developers
- [Rajdeep Bandopadhyay](https://github.com/mareep-raljodid)
