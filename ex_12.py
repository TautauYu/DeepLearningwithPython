import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
dataframe = pandas.read_csv("./data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# Define model
def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim = 13, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal'))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    return model

def larger_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(6, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    return model

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# A Baseline Neural Network Model
# estimator = KerasRegressor(build_fn=baseline_model, epochs = 100, batch_size = 5, verbose =2)
# kfold = KFold(n=len(X), n_folds=10, random_state=seed)

# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" %(results.mean(), results.std()))

estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasRegressor(build_fn=larger_model, epochs = 50, batch_size = 5, verbose = 2)))
pipeline = Pipeline(estimator)
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standarized: %.2f (%.2f) MSE" %(results.mean(), results.std()))