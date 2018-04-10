# -*- coding: utf-8 -*-
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]

# Create model
def SimpleCNNModel():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, input_shape = (32,32,3), data_format = 'channels_last',padding = 'same', activation='relu', W_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, activation = 'relu', W_constraint = maxnorm(3)))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation = 'softmax'))

    # Compile model
    epoch = 25
    learning_rate = 0.01
    decay = learning_rate / epoch
    sgd = SGD(lr = learning_rate, momentum = 0.9, decay = decay, nesterov = False)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    print(model.summary())
    return model

# Create larger model
def larger_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, input_shape = (32,32,3), data_format = 'channels_last', padding = 'same', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation = 'relu', W_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu', W_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_class, activation = 'softmax'))
    # Compile model
    epoch = 25
    learning_rate = 0.01
    decay = learning_rate / epoch
    sgd = SGD(lr = learning_rate, momentum = 0.9, decay = decay, nesterov = 0)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])   
    print(model.summary())
    return model 

cnn_model = SimpleCNNModel()
cnn_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 25, batch_size = 32)

scores = cnn_model.evaluate(X_test, y_test, Verbose = 2)
print("Accuracy: %.2f%%" %(scores[1]*100))