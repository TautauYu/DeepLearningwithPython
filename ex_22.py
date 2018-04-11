# -*- coding: utf-8 -*-
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# Create model
def mlp_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length = max_words))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    print(model.summary())
    return model

def CNN_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length = max_words))
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    return model


# Define model
model = CNN_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 2, batch_size=128, verbose=1)

# Final evalution of the model
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" %(scores[1]*100))