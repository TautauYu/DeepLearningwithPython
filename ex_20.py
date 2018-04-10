# -*- coding: utf-8 -*-
# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [sample][pixels][with][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Define data preparation
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, zca_whitening=True, data_format='channels_first') # ZCA whitening
# datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=180) # Rotation
# datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, width_shift_range=0.5, height_shift_range=1.5) # Shift
# datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, horizontal_flip=True, vertical_flip=False) # Filp

# Fit parameters from data
datagen.fit(X_train)

if not os.path.isdir('images'):
    os.mkdir('images')

# Configure batch size
X_batch, y_batch = datagen.flow(X_train, y_train, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='png').next() # Key point

# Create a grid of 3x3 images
for i in range(0,9):
    plt.subplot(330+1+i)# 3 rows ans 3 cols. E.g. 331: the image is shown in the row 1 and col 1 of the 3x3 grid.
    plt.imshow(X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray'))

# Show the plot
plt.show()