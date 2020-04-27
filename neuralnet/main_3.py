from NeuralNet import NeuralNet, Layer, sigmoid, sigmoid_diff
import pandas
import numpy as np
import matplotlib.pyplot as plt

ss_test = pandas.read_csv('resources/regression/square-large-test.csv', index_col=0)
ss_train = pandas.read_csv('resources/regression/square-large-training.csv', index_col=0)
print(ss_test.iloc[:10, :])

nn = NeuralNet(1)\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(1, lambda act, layer: act, lambda act, layer: 1))\
    .set_optimizer("momentum", 0.1)
nn.train(np.transpose(ss_train.iloc[:, :-1]), np.transpose(ss_train.iloc[:, -1:]), learning_rate=0.002,
         epochs=20, batch_size=10)
print(nn.get_result())

nn2 = NeuralNet(1)\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(5, sigmoid, sigmoid_diff))\
    .add_layer(Layer(1, lambda act, layer: act, lambda act, layer: 1))\
    .set_optimizer("RMSProp", 0.1)
nn2.train(np.transpose(ss_train.iloc[:, :-1]), np.transpose(ss_train.iloc[:, -1:]), learning_rate=0.02,
          epochs=20, batch_size=10, verbose=False)
print(nn2.get_result())

# plot
x = np.arange(21)
plt.plot(x, nn.loss_history, label="momentum")
plt.plot(x, nn2.loss_history, label="RMSProp")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
