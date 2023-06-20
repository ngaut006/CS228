# Making Necessary Imports

import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import torch.nn as nn
from torchvision.models import vgg16
import torch.optim as optim
from tqdm import tqdm

import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

import torchvision.utils as vutils
import matplotlib.pyplot as plt

import os

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from torchsummary import summary

from google.colab import drive

from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import tensorflow as tf
import keras
import os
import shutil

import warnings
warnings.filterwarnings("ignore")

# Load the VGG16 model
vggmodel = vgg16.VGG16()

# Create a new sequential model
newmodel = Sequential()

# Iterate through the layers of the VGG16 model
for i, layer in enumerate(vggmodel.layers):
    if i < 19:  # Include layers up to the 19th layer for feature extraction
        newmodel.add(layer)

# Set all layers in the new model as non-trainable
for layer in newmodel.layers:
    layer.trainable = False

# Generating Colorized Images using saved model weights and Present Dataset
# Loading the model details
model = tf.keras.models.load_model('./colorize_autoencoder_VGG16[3].model',
                                   custom_objects=None,
                                   compile=True)

# Initializing the testpath and fetching images from that folder
testpath = './dataset/dataset/bw_images'
files = os.listdir(testpath)

# Iterate through each file in the testpath
for idx, file in enumerate(files):
    # Load the image and resize it
    test = img_to_array(load_img(testpath+file))
    test = resize(test, (224, 224), anti_aliasing=True)
    test *= 1.0/255

    # Convert the image to LAB color space
    lab = rgb2lab(test)
    l = lab[:, :, 0]

    # Convert L channel to RGB format
    L = gray2rgb(l)
    L = L.reshape((1, 224, 224, 3))

    # Pass the L channel through the newmodel for prediction
    vggpred = newmodel.predict(L)

    # Pass the vggpred through the model for colorization prediction
    ab = model.predict(vggpred)
    ab = ab * 128

    # Create a new LAB image
    cur = np.zeros((224, 224, 3))
    cur[:, :, 0] = l
    cur[:, :, 1:] = ab

    # Convert the LAB image to RGB
    rgb_image = lab2rgb(cur)

    # Convert the RGB image to uint8 format (to cater the need of imsave functions)
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    # Save the colorized image (new empy directory already created to store colorized images while testing)
    imsave('./GenImages'+str(idx)+".jpg", rgb_image_uint8)
    
    
# Visualizing random 64 images from the generated images

# Path to the folder containing your pictures on Google Drive
folder_path = './GenImages'

# Getting a list of all files in the folder
file_list = os.listdir(folder_path)

# Selecting 36 random files from the list
random_files = random.sample(file_list, 36)

# Create a square grid to display the images
fig, axs = plt.subplots(6, 6, figsize=(16, 16))
fig.suptitle('COLORIZED B&W IMAGES', fontsize=16)

# Iterate over the selected files and display the images in the grid
for i, file_name in enumerate(random_files):
    row = i // 6
    col = i % 6
    file_path = os.path.join(folder_path, file_name)
    try:
        img = Image.open(file_path)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
    except Exception as e:
        print(f"Error opening image file {file_name}: {e}")

# Adjust spacing and display the grid
plt.subplots_adjust(hspace=0.05, wspace=0.3)
plt.show()