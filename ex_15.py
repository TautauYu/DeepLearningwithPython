from keras.models import Sequential
from keras.layers import Dense
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
model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
model.add(Dense(8, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

# Compile model
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs = 150, batch_size=10, verbose=2)

# List all data in history
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()