from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy
import pandas

# Function to create model, required for KerasClassifier
def create_model(optimizer = 'rmsprop', init = 'glorot_uniform'):
    # Create model
    model = Sequential()
    model.add(Dense(12, input_dim = 8, init = init, activation = 'relu'))
    model.add(Dense(8, init = init, activation = 'relu'))
    model.add(Dense(1, init = init, activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataset = numpy.loadtxt("./data/pima-indians-diabates.csv", delimiter = ',')

# Split into input (X) and ouput (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = KerasClassifier(build_fn=create_model)

# Grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(optimizer = optimizers, nb_epoch = epochs, batch_size = batches, init = init)
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid_result = grid.fit(X,Y)

# Summarize results
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" %(scores.mean(), scores.std(), params))