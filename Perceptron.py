import math
import numpy as np


def sigmoid(x):
    return (math.exp(-x) + 1)**(-1)


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
        self.weights = None
        self.biases = None
        if activations is None:
            activations = np.repeat(sigmoid, len(layer_sizes) - 1)
        self.activations = activations

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
