from Activations import sigmoid, linear
from NeuralNet import NeuralNet, Layer
from Regularizations import L1_regularization, L2_regularization
from Weights import normal_init
import pandas
import numpy as np
import matplotlib.pyplot as plt


test = pandas.read_csv('resources/regression/multimodal-sparse-test.csv', index_col=0)
train = pandas.read_csv('resources/regression/multimodal-sparse-training.csv', index_col=0)
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]

nn1 = NeuralNet(1, normal_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(5, sigmoid))\
    .add_layer(Layer(1, linear))\
    .set_regularization(L1_regularization.set_params({"coef": 0.05}))
nn1.budget.set_epoch_limit(50).set_detection_limit(1.5)

nn2 = NeuralNet(1, normal_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(5, sigmoid))\
    .add_layer(Layer(1, linear))\
    .set_regularization(L2_regularization.set_params({"coef": 0.05}))
nn2.budget.set_epoch_limit(50).set_detection_limit(1.5)

nn1.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.005, batch_size=8)
nn2.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.005, batch_size=8)

# plot
plt.plot(np.arange(len(nn1.loss_history)), nn1.loss_history, label="L1")
plt.plot(np.arange(len(nn2.loss_history)), nn2.loss_history, label="L2")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Regularization comparison")

# COMMENT:
# Regularization works okay, it seems, overfit detection also, though sometimes the code above has to be run twice.
# Can't really say that one is better than the other, compared with no regularization, however...
