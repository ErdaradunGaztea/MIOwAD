import math
import random
import numpy as np


def sigmoid(x):
    return (math.exp(-x) + 1)**(-1)


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def loss(y, y_hat):
    diff = np.subtract(y_hat, y)
    return sum(sum(np.transpose(diff).dot(diff)))


class InputLayer:
    def __init__(self, result):
        self.result = result


class Layer:
    def __init__(self, size, weights, biases, activation=sigmoid):
        self.result = np.array([])
        self.input_layer = None
        self.nodes = [lambda inp: activation(np.array(weights[index]).dot(inp) + biases[index]) for index in range(size)]

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def compute(self):
        self.result = [list(map(lambda node: node(res), self.nodes)) for res in self.input_layer.result]


class Perceptron:
    def __init__(self, layer_sizes, activations=None):
        self.layers = []
        self.input_size = layer_sizes[0]
        self.hidden_sizes = layer_sizes[1:-1]
        self.output_size = layer_sizes[-1]
        self.sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.weights = []
        self.biases = []
        if activations is None:
            activations = np.repeat(sigmoid, len(layer_sizes) - 1)
        self.activations = activations
        self.learning_step = 10**-3

    def set_parameters(self, weights, biases):
        self.layers.clear()
        self.weights = weights
        self.biases = biases
        self.build()

    def build(self):
        # create hidden layers
        for index, hidden_layer in enumerate(self.hidden_sizes):
            self.layers.append(Layer(hidden_layer, self.weights[index], self.biases[index], self.activations[index]))
        # create output layer
        self.layers.append(Layer(self.output_size, self.weights[-1], self.biases[-1], self.activations[-1]))
        for index, layer in enumerate(self.layers):
            if index > 0:
                layer.set_input_layer(self.layers[index - 1])

    def compute(self, x):
        self.layers[0].set_input_layer(InputLayer(x))
        for layer in self.layers:
            layer.compute()
        return self.layers[-1].result

    def backpropagate(self, x, y):
        i = 0
        y_hat = self.compute(x)
        while not (loss(y, y_hat) < 4 or i > 100):
            print("Iter no. " + str(i))
            print("Loss: " + str(loss(y, y_hat)))
            for index, layer in enumerate(reversed(self.layers)):
                if index == 0:
                    loss_diff = np.subtract(y_hat, y)
                else:
                    loss_diff = np.subtract(y_hat, y)
                    for prev_layer in range(index):
                        loss_diff = loss_diff.dot(np.array(self.weights[prev_layer]))
                    loss_diff = loss_diff.dot(
                        [sigmoid_diff(np.array(self.weights[index]).dot(inp) + self.biases[index]) for inp in self.layers[index - 1].result])
                self.biases[index] = self.biases[index] - self.learning_step * loss_diff
                loss_diff = loss_diff.dot([self.activations[index](np.array(
                        self.weights[index]).dot(inp) + self.biases[index]) for inp in self.layers[index - 1].result])
                self.weights[index] = self.weights[index] - self.learning_step * loss_diff
            y_hat = self.compute(x)
            i += 1
        print("Final loss: " + str(loss(y, y_hat)))
        return None

    def initialize(self, type="zeros"):
        self.layers.clear()
        for index in range(len(self.sizes) - 1):
            if type == "zeros":
                self.weights.append([[0] * self.sizes[index]] * self.sizes[index + 1])
                self.biases.append([0] * self.sizes[index + 1])
            if type == "uniform":
                self.weights.append([[random.uniform(0, 1) for i in range(self.sizes[index])] for j in range(self.sizes[index + 1])])
                self.biases.append([random.uniform(0, 1) for i in range(self.sizes[index + 1])])
        self.build()
