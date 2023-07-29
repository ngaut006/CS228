# Final Project - IMAGE COLORIZATION

This repository contains the code for a VGG16 model implementation for image classification. The code is written in Python and uses the TensorFlow library.

## Contents
The repository contains the following files:

- `model2_vgg16_runnable.py`: This is the main Python file that implements the VGG16 model and provides a runnable script to train and evaluate the model on a given dataset.
- `README.md`: This file provides an overview of the repository and instructions for running the code.

## Dependencies
To run the code, the following dependencies are required:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required dependencies using pip:

*pip install tensorflow keras numpy matplotlib*

## Usage
To use the VGG16 model for image classification, follow these steps:

1. Clone the repository to your local machine:

*git clone https://github.com/ngaut006/CS228.git*

2. Navigate to the project directory:
*python model2_vgg16_runnable.py*

The script will train the VGG16 model on the specified dataset and display the training and validation accuracy over each epoch. After training, it will evaluate the model on the test set and print the test accuracy.

## Dataset

**THE DATASET IS CREATED BY US BY SCRAPPING WEBSITES USING A SELF-BUILT WEB SCRAPER. IN ORDER TO USE THE DATASET, KINDLY REACHOUT TO US VIA E-MAIL**

The code assumes that you have a dataset in a specific format. You need to organize your dataset in separate directories for each class. The directory structure should be as follows:


dataset/

├── class1/

│ ├── image1.jpg

│ ├── image2.jpg

│ └── ...

├── class2/

│ ├── image1.jpg

│ ├── image2.jpg

│ └── ...

└── ...


You need to provide the path to this dataset directory in the `model2_vgg16_runnable.py` file.
