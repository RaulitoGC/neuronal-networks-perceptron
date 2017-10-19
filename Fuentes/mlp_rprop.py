from sklearn import preprocessing, metrics
import numpy as np
from neupy import algorithms, layers, plots

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

# los primero 3067 de los 4601 datos serviran para el entrenmiento
x_train = X[0:3067]
y_train = Y[0:3067]

# el resto de los datos serviran para la validacion (1534)
x_test = X[3067:]
y_test = Y[3067:]

#se crea la red neuronal con la arquitectura 57 -7 -1
rpropnet = algorithms.RPROP(
    [
        layers.Input(57),
        layers.Sigmoid(7),
        layers.Sigmoid(1),
    ],
    error='mse',
    verbose=True,
    shuffle_data=True,
    maxstep=1,
    minstep=1e-7,
)

#se realiza el entrenamiento de la red
rpropnet.train(input_train=x_train,target_train=y_train,epochs=200)

#se muestra un grafico de los errores cometidos en el entrenamiento
plots.error_plot(rpropnet)

y_train_predicted = rpropnet.predict(x_train).round()
y_test_predicted = rpropnet.predict(x_test).round()

# se muestran las predicciones
print(metrics.classification_report(y_train_predicted, y_train))
print(metrics.confusion_matrix(y_train_predicted, y_train))
print()
print(metrics.classification_report(y_test_predicted, y_test))
print(metrics.confusion_matrix(y_test_predicted, y_test))

