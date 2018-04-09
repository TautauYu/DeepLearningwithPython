import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [sample][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters = 30, kernel_size = 5, strides = 1, padding = 'valid', batch_input_shape = (64, 1, 28, 28), data_format = 'channels_first', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 15, kernel_size = 3, strides = 1, padding = 'valid', data_format = 'channels_first', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten()) # 展开成向量形式
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# Build the model
model = CNN_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10, batch_size = 200, verbose = 2)

# Final evalution of the model
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Baseline Error: %.2f%%" %(100-scores[1]*100))