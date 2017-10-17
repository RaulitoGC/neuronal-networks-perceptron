from sklearn import preprocessing, metrics
import numpy as np
from neupy import algorithms, layers

X = []
Y = []

# read the training data
with open('Train.csv') as f:
    for line in f:
        curr = line.split(',')
        new_curr = []
        for item in curr[:len(curr) - 1]:
            new_curr.append(float(item))
        X.append(new_curr)
        Y.append([float(curr[-1])])

X = np.array(X)
X = preprocessing.scale(X) # feature scaling
Y = np.array(Y)

# the first 2500 out of 3000 emails will serve as training data
x_train = X[0:3067]
y_train = Y[0:3067]

# the rest 500 emails will serve as testing data
x_test = X[3067:4601]
y_test = Y[3067:4601]

#rpropnet = algorithms.RPROP((57,7,1))
rpropnet = algorithms.RPROP(
    [
        layers.Input(57),
        layers.Sigmoid(7),
        layers.Sigmoid(1),
    ],
    error='binary_crossentropy',
    verbose=True,
    shuffle_data=True,
    maxstep=1,
    minstep=1e-7,
)
rpropnet.train(input_train=x_train,target_train=y_train,epochs=1000)

y_train_predicted = rpropnet.predict(x_train).round()
y_test_predicted = rpropnet.predict(x_test).round()

print(metrics.classification_report(y_train_predicted, y_train))
print(metrics.confusion_matrix(y_train_predicted, y_train))
print()
print(metrics.classification_report(y_test_predicted, y_test))
print(metrics.confusion_matrix(y_test_predicted, y_test))

