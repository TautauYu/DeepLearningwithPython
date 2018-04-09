import pandas
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

#Learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("./data/ionosphere.csv", header=None)
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Create model
model = Sequential()
model.add(Dense(34, input_dim = 34, init = 'normal', activation = 'relu'))
model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

# Compile model
"""
# Time-Based Learning Rate Schedule
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)

model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Fit model
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28)
"""
# Drop-Based Learning Rate Schedule
sgd = SGD(lr =0.0, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

lrate = LearningRateScheduler(step_decay) # step_decay是以epoch为参数的函数，epoch从0开始，返回一个新的浮点型学习率
callbacks_list = [lrate]

# Fit model
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list)