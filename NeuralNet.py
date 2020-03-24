import math
import numpy as np


def sigmoid(x):
    return (math.exp(-x) + 1) ** (-1)


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def split(data, batch_size):
    return [np.transpose(batch) for batch in np.split(np.transpose(data), range(batch_size, data.shape[1], batch_size))]


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.values = None

    def set_data(self, data):
        self.values = data

    def get_values(self):
        return self.values


class Layer:
    def __init__(self, size, activation, activation_diff):
        # layer parameters
        self.size = size
        self.activations = np.array([np.vectorize(activation)] * size)
        self.activation_diffs = np.array([np.vectorize(activation_diff)] * size)
        self.input = None
        self.weights = None
        self.biases = None
        # computed values
        self.weighted_input = None
        self.values = None
        # backpropagation
        self.local_gradient = None
        self.weights_diff = None
        self.biases_diff = None
        # optimizer
        self.optimizer = None
        self.momentum_coef = 0
        self.weights_momentum = None
        self.biases_momentum = None

    def add_input(self, inp, weight_init):
        self.input = inp
        if weight_init == "uniform":
            self.weights = np.array([[np.random.uniform(0, 1) for _ in range(inp.size)] for _ in range(self.size)])
            self.biases = np.array([[np.random.uniform(0, 1)] for _ in range(self.size)])
        elif weight_init == "normal":
            self.weights = np.array([[np.random.normal(0, 1) for _ in range(inp.size)] for _ in range(self.size)])
            self.biases = np.array([[np.random.normal(0, 1)] for _ in range(self.size)])
        self.weights_momentum = np.array([[0 for _ in range(inp.size)] for _ in range(self.size)])
        self.biases_momentum = np.array([[0] for _ in range(self.size)])

    def predict(self):
        self.weighted_input = np.array(self.weights.dot(self.input.values)) + np.array(self.biases)
        self.values = np.array([activation(value) for activation, value in zip(self.activations, self.weighted_input)])

    def backpropagate(self, weighted_error=None):
        self.local_gradient = np.multiply(np.array([activation_diff(value) for activation_diff, value in
                                                    zip(self.activation_diffs, self.weighted_input)]), weighted_error)
        if not self.optimizer:
            self.weights_diff = self.local_gradient.dot(np.transpose(self.input.values))
            self.biases_diff = self.local_gradient.dot(np.ones((self.input.values.shape[1], 1)))
        elif self.optimizer == "momentum":
            self.weights_momentum = self.local_gradient.dot(np.transpose(self.input.values)) \
                                    - self.weights_momentum * self.momentum_coef
            self.biases_momentum = self.local_gradient.dot(np.ones((self.input.values.shape[1], 1))) \
                                   - self.biases_momentum * self.momentum_coef
            self.weights_diff = self.weights_momentum
            self.biases_diff = self.biases_momentum
        elif self.optimizer == "RMSProp":
            self.weights_diff = self.local_gradient.dot(np.transpose(self.input.values))
            self.biases_diff = self.local_gradient.dot(np.ones((self.input.values.shape[1], 1)))
            self.weights_momentum = self.weights_momentum * self.momentum_coef + \
                                    self.weights_diff**2 * (1 - self.momentum_coef)
            self.biases_momentum = self.biases_momentum * self.momentum_coef + \
                                   self.biases_diff**2 * (1 - self.momentum_coef)
            self.weights_diff = self.weights_diff / (self.weights_momentum**0.5)
            self.biases_diff = self.biases_diff / (self.biases_momentum**0.5)


class NeuralNet:
    def __init__(self, size, weight_init="uniform"):
        self.layers = [InputLayer(size)]
        self.loss_history = None
        self.weight_init = weight_init

    def add_layer(self, layer):
        layer.add_input(self.layers[-1], self.weight_init)
        self.layers.append(layer)
        return self

    def predict(self, data):
        self.layers[0].set_data(data)
        for layer in self.layers[1:]:
            layer.predict()

    def backpropagate(self, y, learning_rate=0.001):
        self.layers[-1].backpropagate(self.get_result() - y)
        for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
            weighted_error = np.transpose(next_layer.weights).dot(next_layer.local_gradient)
            layer.backpropagate(weighted_error)
        for layer in self.layers[1:]:
            layer.weights -= learning_rate * layer.weights_diff
            layer.biases -= learning_rate * layer.biases_diff

    def get_result(self):
        return self.layers[-1].values

    def get_loss(self, y):
        return np.mean(np.transpose(self.get_result() - y) ** 2)

    def set_optimizer(self, optimizer, coefficient):
        for layer in self.layers[1:]:
            layer.optimizer = optimizer
            if optimizer == "momentum" or optimizer == "RMSProp":
                layer.momentum_coef = coefficient
        return self

    def train(self, data, y, epochs=1000, learning_rate=0.001, batch_size=10, max_loss=0.1, verbose=True):
        self.loss_history = []
        epoch = 0
        self.predict(data)
        loss = self.get_loss(y).array[0]
        self.loss_history.append(loss)
        while epoch < epochs and loss > max_loss:
            print("Epoch: {}".format(epoch + 1))
            batches = zip(split(data, batch_size), split(y, batch_size))
            for index, batch in enumerate(batches):
                self.predict(batch[0])
                self.backpropagate(batch[1], learning_rate)
                self.predict(data)
                loss = self.get_loss(y).array[0]
                if verbose:
                    print("Batch {0}/{1}".format(index + 1, math.ceil(y.shape[1] / batch_size)))
                    print("Loss: {}".format(loss))
            print("==========================")
            self.loss_history.append(loss)
            epoch += 1
