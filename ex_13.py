from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import numpy
import os

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter = ",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
model.add(Dense(8, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model
model.fit(X, Y, epochs = 150, batch_size = 10, verbose = 2)

# Evaluate the model
scores = model.evaluate(X, Y, verbose = 2)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

"""
# Serialize model to JSON
model_json = model.to_json()
with open("./SaveModel/model.json", "w") as json_file:
    json_file.write(model_json)
"""

# Serialize model to YAML
model_yaml = model.to_yaml()
with open("./SaveModel/model.yaml","w") as yaml_file:
    yaml_file.write(model_yaml)

# Serialize weights to HDF5
model.save_weights("./SaveModel/model.h5")
print("Saved model to disk")

"""
# Load json and create model
json_file = open("./SaveModel/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
"""

# Load yaml and create model
yaml_file = open("./SaveModel/model.yaml","r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)

# Load weights into new model
loaded_model.load_weights("./SaveModel/model.h5")
print("Loaded model from disk")

# Evaluate loaded model on the test data
loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
scores = loaded_model.evaluate(X, Y, verbose = 2)
print("%s: %.2f%%" %(loaded_model.metrics_names[1], scores[1]*100))