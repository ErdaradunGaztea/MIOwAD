"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from kohonen.LearningRestraints import learning_exp


def progress(value, length):
    """Creates and overwrites a progress bar with a length of 50 signs and percentage value."""
    prgs = int(100 * value / length)
    sys.stdout.write("\rProgress: [{0}] {1}%".format("#" * (prgs // 2) + " " * (50 - prgs // 2), prgs))
    sys.stdout.flush()


class KohonenNet:
    def __init__(self):
        """Main class of the package, creates Kohonen network object."""
        self.topology = None
        self.neighbourhood = None
        self.learning_restraint = learning_exp.set_params({"coef": 5})
        self.task = None
        # overwrite __update_weights__ with vectorized version
        self.__update_weights__ = np.vectorize(self.__update_weights__, excluded=['bmu', 'row', 't'])

    def __update_weights__(self, node, bmu, row, t):
        """Used during training to update weights of nodes, shouldn't be used as standalone call."""
        # compute impact from distance in neuron topology and impact from learning time
        coefficient = self.neighbourhood.get_impact(node, bmu) * self.learning_restraint.restrain(t)
        # update weights from impact and weight difference
        node.weights += coefficient * (row - node.weights)

    def add_topology(self, topology):
        self.topology = topology
        # adds topology to neighbourhood in case neighbourhood was added already
        if self.neighbourhood:
            self.neighbourhood.set_topology(topology)
        return self

    def add_neighbourhood(self, neighbourhood):
        self.neighbourhood = neighbourhood.set_topology(self.topology)
        return self

    def add_task(self, task):
        self.task = task
        return self

    def train(self, runs=50):
        """Initializes weights and trains the network."""
        self.topology.__init_weights__(self.task)
        data_length = len(self.task.get_x())
        # generate plot with initial node distribution
        self.plot()
        for t in range(runs):
            # generate random order of rows
            data = self.task.get_x().sample(frac=1).reset_index(drop=True)
            # for each row find BMU (best matching unit)
            for i, row in data.iterrows():
                # display progress bar
                progress(t * data_length + i + 1, runs * data_length)
                # create vectorized distance function for current row
                distance_func = np.vectorize(lambda node: np.linalg.norm(node.weights - row))
                # find BMU using given distance_func
                bmu = self.topology.nodes[np.argmin(distance_func(self.topology.nodes))]
                # another vectorized function to update weights
                self.__update_weights__(self.topology.nodes, bmu=bmu, row=row, t=t)
            # generate plot with current node distribution
            self.plot(iteration=t+1)
        return self

    def plot(self, iteration=0):
        """Creates and writes .png file for first two dimensions of training dataset and trained nodes."""
        # scatter training data
        plt.scatter(self.task.get_x().iloc[:, 0],
                    self.task.get_x().iloc[:, 1],
                    label="training data")
        # scatter trained weights
        plt.scatter([node.weights[0] for node in self.topology.nodes],
                    [node.weights[1] for node in self.topology.nodes],
                    label="node weights",
                    color='red')
        # add plot description
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Node weights vs training data, iter {0}".format(iteration))
        # save and close
        plt.savefig("KohonenNet{0}.png".format(iteration))
        plt.close()
