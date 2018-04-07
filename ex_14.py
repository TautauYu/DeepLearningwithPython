from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim = 8, init = 'iuniform', activation = 'relu'))
model.add(8, init = 'uniform', activation = 'relu')
model.add(1, init = 'uniform', activation = 'sigmoid')

"""
# Load wieights
model.load_weights("weights.best.hdf5")
print("Create model and loaded weightd from file")
"""

# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Checkpoint
filepath = "weight.bset.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)

"""
scores = model.evaluate(X, Y, verbose=2)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
"""