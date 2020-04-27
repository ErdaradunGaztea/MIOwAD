"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np


class Neighbourhood:
    def __init__(self, impact):
        """Computes impact of distance on weight change."""
        self.radius = 1
        self.__impact__ = impact
        self.topology = None

    def set_radius(self, radius):
        """Updates radius of neighbourhood, parameter which scales distance between nodes."""
        # typical usage is `dist(node0, node1) / radius`
        self.radius = radius
        return self

    def set_topology(self, topology):
        """Updates topology reference. Should be done automatically be KohonenNet object."""
        self.topology = topology
        return self

    def get_impact(self, node0, node1):
        """Computes impact of distance between nodes on weight change value."""
        return self.__impact__(self.topology.distance(node0, node1), self.radius)


# names should be self-explaining
neighbour_gauss = Neighbourhood(
    lambda dist, r: np.exp(-dist / r)
)
neighbour_mexican_hat = Neighbourhood(
    lambda dist, r: (2 - 4 * np.square(dist)) * np.exp(-dist / r)
)
