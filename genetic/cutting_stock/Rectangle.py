"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import copy
import itertools

import numpy as np


class Rectangle:
    def __init__(self, template, pos):
        self.id = id(template)
        self.size = copy.deepcopy(template.size)
        self.value = template.value
        self.pos = np.array(pos)

    def bottom_left(self):
        """Returns coordinates of multidimensional \"bottom left\" vertex (with lowest coords values)."""
        return self.pos

    def top_right(self):
        """Returns coordinates of multidimensional \"top right\" vertex (with highest coords values)."""
        return self.pos + self.size

    def overlap(self, r2):
        """Checks if both rectangles overlap."""
        return np.all(self.bottom_left() < r2.top_right()) and np.all(r2.bottom_left() < self.top_right())

    def collide(self, r2, axis=0):
        """Checks if both rectangles would collide, if one would slide indefinitely along given axis."""
        return (self.pos[axis] < r2.pos[axis] + r2.size[axis]) and (r2.pos[axis] < self.pos[axis] + self.size[axis])

    def vertices(self):
        """Returns list of coordinates of rectangle vertices (4 coordinates for 2D rectangle)."""
        return [self.pos + np.array(i) for i in itertools.product(*zip(np.zeros_like(self.size), self.size))]

    def rotate_randomly(self):
        """Rotates rectangle by reshuffling its sizes (thus leaving bottom left vertex in place)."""
        np.random.shuffle(self.size)
        return self
