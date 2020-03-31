from NeuralNet import NeuralNet, Layer, sigmoid, sigmoid_diff, softmax, softmax_diff
import pandas
import numpy as np
import matplotlib.pyplot as plt


def one_hot(df, column):
    dummies = pandas.get_dummies(df[column], prefix=column)
    df_without_column = df.drop([column], axis=1)
    return df_without_column, dummies


# ---- EASY DATASET ----
# read data
test = pandas.read_csv('resources/classification/easy-test.csv')
train = pandas.read_csv('resources/classification/easy-training.csv')
# one-hot-encode class column
x_test, y_test = one_hot(test, 'c')
x_train, y_train = one_hot(train, 'c')

# create two neural networks with different activations in output layer
nn = NeuralNet(2)\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(2, lambda act, layer: act, lambda act, layer: 1))
nn2 = NeuralNet(2)\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(2, softmax, softmax_diff))

# train both networks with same parameters
nn.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.02, epochs=30, batch_size=20)
nn2.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.02, epochs=30, batch_size=20)

# plot
x = np.arange(30)
plt.plot(x, nn.loss_history, label="identity")
plt.plot(x, nn2.loss_history, label="softmax")
plt.xlabel("epoch")
plt.ylabel("MSE")
# plt.yscale("log")
plt.legend()
plt.title("Identity and softmax MSE comparison")

# COMMENT:
# It seems like softmax works better, identity doesn't drop equally fast.
# Tested it for 20 and 30 epochs (well, actually more than only these two) and the results are quite consistent.
# For honest, though, one run gave me identity network staying with its MSE at 0.9 after 20 epochs.
# Softmax improved its MSE in every run so far.

# ---- ANOTHER DATASET ----
# code below is actually almost copy-pasted, because it works and it's not really something worth making a function
# read data
test = pandas.read_csv('resources/classification/rings3-regular-test.csv')
train = pandas.read_csv('resources/classification/rings3-regular-training.csv')
# one-hot-encode class column
x_test, y_test = one_hot(test, 'c')
x_train, y_train = one_hot(train, 'c')

# create two neural networks with different activations in output layer
nn = NeuralNet(2)\
    .add_layer(Layer(6, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(3, lambda act, layer: act, lambda act, layer: 1))
nn2 = NeuralNet(2)\
    .add_layer(Layer(6, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(3, softmax, softmax_diff))

# train both networks with same parameters
nn.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.02, epochs=20, batch_size=30)
nn2.train(np.transpose(x_train), np.transpose(y_train), learning_rate=0.02, epochs=20, batch_size=30)

# plot
x = np.arange(20)
plt.plot(x, nn.loss_history, label="identity")
plt.plot(x, nn2.loss_history, label="softmax")
plt.xlabel("epoch")
plt.ylabel("MSE")
# plt.yscale("log")
plt.legend()
plt.title("Identity and softmax MSE comparison")

# COMMENT:
# Softmax is still better, though there isn't THAT much difference and more epochs do very little.
