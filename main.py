from Perceptron import Perceptron, sigmoid
import pandas
import matplotlib.pyplot as plt

p1 = Perceptron([1, 5, 1], activations=[sigmoid, lambda x: x])
p2 = Perceptron([1, 10, 1])
p3 = Perceptron([1, 5, 5, 1])

ss_train = pandas.read_csv('resources/regression/square-simple-training.csv', index_col=0)
ss_test = pandas.read_csv('resources/regression/square-simple-test.csv', index_col=0)

p1.set_parameters([[[1], [-2], [4], [-4], [10]], [[-10, 100, -15, 25, 17]]], [[5, 0.5, -2, 2, 0], [0]])
res = p1.compute([[x] for x in ss_train.x])

plt.scatter(ss_train.x, ss_train.y)
plt.scatter(ss_train.x, res)
