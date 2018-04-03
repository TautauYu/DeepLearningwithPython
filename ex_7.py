# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# Load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv",delimiter=",")
# Split into input (x) and ouput (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# Create model
model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
model.add(Dense(8, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
# Compile model
model.Compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch = 150, batch_size = 10)
# Evaluate the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))