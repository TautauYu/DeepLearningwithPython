import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fix randam seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("../data/sonar.csv", header=None)
dataset = dataframe.values

# Split into input (X) output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# Encode class values as inetgers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y) # Generating the 0-1 label

# Define and Compile Baseline Model
def create_basedline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

# Evaluate model with standarized dataset
estimator = KerasClassifier(build_fn=create_basedline, nb_epoch=100, batch_size=5, verbose=2)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))