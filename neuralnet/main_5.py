from Activations import ReLU, sigmoid, linear, tanh
from NeuralNet import NeuralNet, Layer
import pandas
import numpy as np
import matplotlib.pyplot as plt


def one_hot(df, column):
    dummies = pandas.get_dummies(df[column], prefix=column)
    df_without_column = df.drop([column], axis=1)
    return df_without_column, dummies


# ---- REGRESSION ----
# read data
test = pandas.read_csv('resources/regression/multimodal-large-test.csv', index_col=0)
train = pandas.read_csv('resources/regression/multimodal-large-training.csv', index_col=0)
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]

# create four neural networks with different activations in hidden layers
nn = NeuralNet(1)\
    .add_layer(Layer(5, sigmoid))\
    .add_layer(Layer(5, sigmoid))\
    .add_layer(Layer(1, linear))
nn2 = NeuralNet(1)\
    .add_layer(Layer(5, ReLU))\
    .add_layer(Layer(5, ReLU))\
    .add_layer(Layer(1, linear))
nn3 = NeuralNet(1)\
    .add_layer(Layer(5, tanh))\
    .add_layer(Layer(5, tanh))\
    .add_layer(Layer(1, linear))
nn4 = NeuralNet(1)\
    .add_layer(Layer(5, linear))\
    .add_layer(Layer(5, linear))\
    .add_layer(Layer(1, linear))

# train both networks with same parameters
nn.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.00005, epochs=25, batch_size=200)
nn2.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.002, epochs=25, batch_size=200)
nn3.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.00005, epochs=25, batch_size=200)
nn4.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.000002, epochs=25, batch_size=200)

# plot
x = np.arange(25)
plt.plot(x, nn.loss_history, label="sigmoid")
plt.plot(x, nn2.loss_history, label="ReLU")
plt.plot(x, nn3.loss_history, label="tanh")
plt.plot(x, nn4.loss_history, label="linear")
plt.xlabel("epoch")
plt.ylabel("MSE")
# plt.yscale("log")
plt.legend()
plt.title("Hidden activations MSE comparison")

# COMMENT:
# I had to set very small learning rates to have algorithm not diverge.
# ReLU was basicly useless, couldn't get learning rate that would work for it.
# Linear activation worked okay-ish, converged quickly and then just said: "lol, my job's done".
# Sigmoid had simply beautiful MSE convergence, but quite slow. Bigger learning rate wasn't helpful, though.
# Lastly, tanh. Maybe could use a bit smaller learning rate,
# but I think it's clear it has more potential there than sigmoid... probably.


# ---- MIXED ACTIVATION ----
# It's possible to mix activations within one layer, but it has to be manually accessed (it's safer this way).
nn5 = NeuralNet(1)\
    .add_layer(Layer(5, sigmoid))\
    .add_layer(Layer(5, tanh))\
    .add_layer(Layer(1, linear))
nn5.layers[-2].activations[2] = tanh
nn5.layers[-2].activations[3] = ReLU
nn5.layers[-2].activations[4] = linear
nn5.layers[-2].activations[2] = sigmoid
nn5.layers[-2].activations[3] = ReLU
nn5.layers[-2].activations[4] = linear
nn5.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.00001, epochs=25, batch_size=200)

# plot
x = np.arange(25)
plt.plot(x, nn.loss_history, label="sigmoid")
plt.plot(x, nn3.loss_history, label="tanh")
plt.plot(x, nn5.loss_history, label="mixed")
plt.xlabel("epoch")
plt.ylabel("MSE")
# plt.yscale("log")
plt.legend()
plt.title("Mixed activation compared to standard ones")

# It actually has quite nice results, comparable with pure sigmoid or tanh.
# It's a nice alternative to creating complicated architectures with several parallel layers or something.
