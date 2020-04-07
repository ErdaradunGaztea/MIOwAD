import math
import numpy as np


def split(data, batch_size):
    """Splits data into batches."""
    # implemented because of that transposition; should I get rid of it?
    return [np.transpose(batch) for batch in np.split(np.transpose(data), range(batch_size, data.shape[1], batch_size))]


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.values = None

    def set_data(self, data):
        self.values = data


class Layer:
    def __init__(self, size, activation):
        # layer parameters
        self.size = size
        self.input = None
        self.weights = None
        self.biases = None
        # activations
        # setting it like that allows manually setting different activations for different neurons within layer
        self.activations = np.array([activation] * size)
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

    def __add_input__(self, inp, weight_init):
        """Passes reference to input Layer object and initializes weights."""
        self.input = inp
        if weight_init == "uniform":
            self.weights = np.array([[np.random.uniform(0, 1) for _ in range(inp.size)] for _ in range(self.size)])
            self.biases = np.array([[np.random.uniform(0, 1)] for _ in range(self.size)])
        elif weight_init == "normal":
            self.weights = np.array([[np.random.normal(0, 1) for _ in range(inp.size)] for _ in range(self.size)])
            self.biases = np.array([[np.random.normal(0, 1)] for _ in range(self.size)])
        self.weights_momentum = np.array([[0 for _ in range(inp.size)] for _ in range(self.size)])
        self.biases_momentum = np.array([[0] for _ in range(self.size)])

    def __predict__(self):
        # compute weighted input (input activations + biases), without applying activation function
        self.weighted_input = np.array(self.weights.dot(self.input.values)) + np.array(self.biases)
        # initialize returned values with zeros
        self.values = np.zeros(self.weighted_input.shape)
        # apply activation function to neuron
        # iterate over observations
        for i in range(self.weighted_input.shape[1]):
            # iterate over neurons within layer
            for j in range(self.weighted_input.shape[0]):
                # apply neuron activation function to activation values of this neuron and whole layer
                self.values[j, i] = self.activations[j].get_function()(self.weighted_input[j, i], self.weighted_input[:, i])
        # NOTE: there used to be vectorized function, but passing whole layer made it too complicated

    def __backpropagate__(self, weighted_error=None):
        """Makes some mathematical magic and possibly generates uncatched computational errors."""
        values = np.zeros(self.weighted_input.shape)
        # apply derivative of activation function to neuron
        # iterate over observations
        for i in range(self.weighted_input.shape[1]):
            # iterate over neurons within layer
            for j in range(self.weighted_input.shape[0]):
                # apply derivative of neuron activation function to activation values of this neuron and whole layer
                values[j, i] = self.activations[j].get_derivative()(self.weighted_input[j, i], self.weighted_input[:, i])
        self.local_gradient = np.multiply(values, weighted_error)
        # OPTIMIZATION AREA
        # only looks scary, but it's simple in fact
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
        """Appends layer to NeuralNet object. Should be given Layer object as parameter."""
        layer.__add_input__(self.layers[-1], self.weight_init)
        self.layers.append(layer)
        return self

    def predict(self, data):
        """Uses internal weights to predict answer for given data."""
        self.layers[0].set_data(data)
        for layer in self.layers[1:]:
            layer.__predict__()

    def __backpropagate__(self, y, learning_rate=0.001):
        # compute weight and bias changes for every layer starting with the last
        self.layers[-1].__backpropagate__(self.get_result() - y)
        for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
            weighted_error = np.transpose(next_layer.weights).dot(next_layer.local_gradient)
            layer.__backpropagate__(weighted_error)
        # apply computed changes to weights and biases
        # again, starting with the last layer (I forgot why, seems like it doesn't matter)
        for layer in self.layers[1:]:
            layer.weights -= learning_rate * layer.weights_diff
            layer.biases -= learning_rate * layer.biases_diff

    def get_result(self):
        """Returns last predicted values."""
        return self.layers[-1].values

    def get_loss(self, y):
        return np.mean(np.transpose(self.get_result() - y) ** 2)

    def set_optimizer(self, optimizer, coefficient):
        """Adds optimizer to backpropagation algorithm. Passed as string, either \"momentum\" or \"RMSProp\".
        Also takes coefficient value."""
        for layer in self.layers[1:]:
            layer.optimizer = optimizer
            if optimizer == "momentum" or optimizer == "RMSProp":
                layer.momentum_coef = coefficient
        return self

    def train(self, data, y, epochs=1000, learning_rate=0.001, batch_size=10, verbose=True):
        """Calls backpropagation algorithm with MSE loss function to fit weights."""
        # initialize some variables
        self.loss_history = np.zeros(epochs)
        epoch = 0
        # initial prediction
        self.predict(data)
        loss = self.get_loss(y).sum()
        while epoch < epochs:
            print("Epoch: {}".format(epoch + 1))
            # split data into batches
            batches = zip(split(data, batch_size), split(y, batch_size))
            for index, batch in enumerate(batches):
                # make prediction for given batch
                self.predict(batch[0])
                # run backpropagation with given batch
                self.__backpropagate__(batch[1], learning_rate)
                # compute loss for whole dataset
                self.predict(data)
                loss = self.get_loss(y).sum()
                if verbose:
                    print("Batch {0}/{1}".format(index + 1, math.ceil(y.shape[1] / batch_size)))
                    print("Loss: {}".format(loss))
            print("==========================")
            # save final loss for given epoch
            self.loss_history[epoch] = loss
            epoch += 1
