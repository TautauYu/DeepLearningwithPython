import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pandas.read_csv("./data/iris.csv", header = None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Encode class values as intergers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert intergers to dummy variables (i.e. one-hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Define baseline model
def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(4, input_dim = 4, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(3, kernel_initializer = 'normal', activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# Create wrapper for neural network model for using in scikit-learn
estimator = KerasClassifier(build_fn = baseline_model, nb_epoch = 200, batch_size = 5, verbose = 2)

# K fold
kfold = KFold(n = len(X), n_folds = 10, shuffle = True, random_state = seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))