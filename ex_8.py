# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy
# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# Load pima indians dataset
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv",delimiter=",")
# Split into input (x) and ouput (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# Define 10-fold cross validation test harness
kfold = StratifiedKFold(y = Y, n_folds = 10, shuffle = True, random_state = seed)
cvscores = []
for i, (train, test) in enumerate(kfold):
    # Create model
    model = Sequential()
    model.add(Dense(12, input_dim = 8, init = 'uniform', activation = 'relu'))
    model.add(Dense(8, inti = 'uniform', activation = 'relu'))
    model.add(Dense(1, inti = 'uniform', activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrcs = ['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], nb_epoch = 150, batch_size = 10, verbose = 0)
    # Evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose = 0)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)
print("%.2f%% (+/- %.2f%%)" %(numpy.mean(cvscores),numpy.std(cvscores)))