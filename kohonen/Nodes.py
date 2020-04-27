"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""


class Node:
    def __init__(self, position):
        # dictionary containing position of node within Topology (usually with x, y and z entries)
        self.pos = position
        # list of weights corresponding to each column of input
        self.weights = None
