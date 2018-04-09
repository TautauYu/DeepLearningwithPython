import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("./data/ionosphere.csv", header=None)
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:,0:34]
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
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Fit model
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28)