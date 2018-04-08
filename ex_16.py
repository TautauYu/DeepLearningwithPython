import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv('./data/sonar.csv', header=None)
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# Encode class values as intergers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Create model
# Using dropout on the visible layer
def create_baseline_dropoutOnV():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,))) # Using dropout on the visible layer
    model.add(Dense(60, init= 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dense(30, init = 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

    # Compile model
    sgd = SGD(lr = 0.1, momentum=0.9, decay = 0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Using dropout on the hidden layer

def create_baseline_dropoutOnH():
    model = Sequential()
    model.add(Dense(60, input_dim = 60, init= 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, init = 'normal', activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

    # Compile model
    sgd = SGD(lr = 0.1, momentum=0.9, decay = 0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline_dropoutOnH, epochs = 300, batch_size = 16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))
