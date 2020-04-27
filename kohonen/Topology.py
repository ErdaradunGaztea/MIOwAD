"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np

from kohonen.Nodes import Node


"""
Important note!
Class below is mainly point of interest of the second exercise. For now the knowledge that it can be used to
initialize rectangular network topology should suffice. These rectangles are kept as 1D list of pairs:
(list_of_weights, position). Also, weight initialization is explained below.
"""


def __initialize_weights__(node, stats):
    """Generates vector of weights from normal distribution. Generated numbers are drawn from independent distributions
    with means and standard deviations of each column."""
    # would love a lambda, but had to assign values to field of each node
    node.weights = np.random.normal(stats.loc['mean'], stats.loc['std'])


class Topology:
    def __init__(self, distance, instantiate):
        """Interface. Contains 1D list of nodes and a method used to convert positions of two nodes
        into distance value between them."""
        # why initialize with empty array if we can create full array later?
        self.nodes = None
        self.__distance__ = distance
        self.__instantiate__ = instantiate
        # we vectorize weight-initializing function over nodes (thus excluding stats from vectorization)
        self.__weights_func__ = np.vectorize(__initialize_weights__, excluded=['stats'])

    def __init_weights__(self, task):
        """Computes mean and variance to initialize weights from normal distribution."""
        stats = task.get_x().describe()
        # stats have to be a named parameter for vectorization exclusion to work properly
        self.__weights_func__(self.nodes, stats=stats)

    def instantiate(self, dimensions):
        """Creates nodes according to topology type and passed dimensions."""
        self.nodes = self.__instantiate__(dimensions)
        return self

    def distance(self, node0, node1):
        """Returns distance value between two nodes."""
        return self.__distance__(node0, node1)


def __rectangle_inst__(dimensions):
    """Creates 1D list of neurons where each neuron has its 2D coordinates."""
    nodes = np.empty(dimensions[0] * dimensions[1], dtype=Node)
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            nodes[x + y * dimensions[0]] = Node({"x": x, "y": y})
    return nodes


"""Creates a rectagonal network, where distances between nodes are computed using euclidean distance."""
rectangle_eucl = Topology(
    lambda node0, node1: np.sqrt(
        np.square(node1.pos['x'] - node0.pos['x']) + np.square(node1.pos['y'] - node0.pos['y'])),
    __rectangle_inst__
)
"""Creates a rectagonal network, where distances between nodes are computed using shortest path distance."""
rectangle = Topology(
    lambda node0, node1: np.abs(node1.pos['x'] - node0.pos['x']) + np.abs(node1.pos['y'] - node0.pos['y']),
    __rectangle_inst__
)

# some of these below will be created as a part of second exercise
# hexagon = None
# hexagonal_rectangle = None
# toroidal_rectangle = None
