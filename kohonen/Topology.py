"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np

from kohonen.Nodes import Node


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


def __hexagonal_rectangle_inst__(dimensions):
    """Creates 1D list of neurons where each neuron has its 3D coordinates."""
    nodes = np.empty(dimensions[0] * dimensions[1], dtype=Node)
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            # the idea is to assign 3 coordinates (yes, one of them is redundant)
            # "x" is the position on 150 degree axis
            # "y" stays as it is (90 degree axis)
            # "z" is the position on 30 degree axis
            nodes[x + y * dimensions[0]] = Node({"x": x + (y + 1) // 2,
                                                 "y": y,
                                                 "z": x + (dimensions[1] - 1) // 2 - y // 2})
    return nodes


def __hexagon_inst__(dimensions):
    """Creates 1D list of neurons where each neuron has its 3D coordinates. Dimensions should be passed as follows:
    [top_edge_length, left_edge_length, right_edge_length]."""
    nodes = np.empty(dimensions[0] * (dimensions[1] - 1) +
                     dimensions[1] * (dimensions[2] - 1) +
                     dimensions[2] * (dimensions[0] - 1) + 1, dtype=Node)
    nodes_created = 0
    # x by y-1 rectangle
    for x in range(dimensions[0]):
        for y in range(dimensions[1] - 1):
            nodes[x + y * dimensions[0]] = Node({"x": x, "y": y, "z": dimensions[1] - 1 - y + x})
    nodes_created += dimensions[0] * (dimensions[1] - 1)
    # y by z-1 rectangle
    for y in range(dimensions[1]):
        for z in range(dimensions[2] - 1):
            nodes[nodes_created + y + z * dimensions[1]] = Node({"x": dimensions[2] - 1 - z + y, "y": y, "z": z})
    nodes_created += dimensions[1] * (dimensions[2] - 1)
    # z by x-1 rectangle
    for z in range(dimensions[2]):
        for x in range(dimensions[0] - 1):
            nodes[nodes_created + z + x * dimensions[2]] = Node({"x": x, "y": dimensions[0] - 1 - x + z, "z": z})
    nodes_created += dimensions[2] * (dimensions[0] - 1)
    # central node
    nodes[nodes_created] = Node({"x": dimensions[0] - 1, "y": dimensions[1] - 1, "z": dimensions[0] - 1})
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
"""Creates a rectagonal network with shortest path distance, where nodes are organized in hexagon net."""
hexagonal_rectangle = Topology(
    lambda node0, node1: np.max([
        np.abs(node1.pos['x'] - node0.pos['x']),
        np.abs(node1.pos['y'] - node0.pos['y']),
        np.abs(node1.pos['z'] - node0.pos['z'])
    ]),
    __hexagonal_rectangle_inst__
)
"""Creates a hexagonal network with shortest path distance."""
hexagon = Topology(
    lambda node0, node1: np.max([
        np.abs(node1.pos['x'] - node0.pos['x']),
        np.abs(node1.pos['y'] - node0.pos['y']),
        np.abs(node1.pos['z'] - node0.pos['z'])
    ]),
    __hexagon_inst__
)
