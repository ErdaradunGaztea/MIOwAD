"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import random

import numpy as np


class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.rectangles = []
        # number of dimensions (better not change it)
        self.axes = 2

    def insert_rectangle(self, rectangle):
        # first remove any conflicting rectangles
        for index in reversed(range(len(self.rectangles))):
            if rectangle.overlap(self.rectangles[index]):
                del self.rectangles[index]
        # then append inserted rectangle
        self.rectangles.append(rectangle)
        return self

    def flip(self, axis=0):
        for rectangle in self.rectangles:
            rectangle.pos[axis] = -rectangle.pos[axis] - rectangle.size[axis]

    def random_push(self, axis=0):
        # sort rectangles by their front so that we limit computations
        self.rectangles.sort(key=lambda rect: rect.pos[axis])
        for index in range(len(self.rectangles)):
            # the more rectangles, the smaller probability of mutation for single rectangle
            # 10 is simply frequency, seems reasonable, but may be tweaked
            if random.random() < np.exp(-len(self.rectangles) / 10):
                if index > 0:
                    # now sort preceding rectangles by their back (but only those that collide)
                    limits = [rect.pos[axis] + rect.size[axis] for rect in self.rectangles[:index-1]
                              if self.rectangles[index].collide(rect, axis)]
                    if len(limits) > 0:
                        self.rectangles[index].pos[axis] = random.uniform(max(limits), self.rectangles[index].pos[axis])
                        continue
                # if there are no colliders (or this was first rectangle) push it all the way down
                lower_limit = -np.sqrt(self.radius ** 2 - (self.rectangles[index].size[axis] / 2) ** 2)
                self.rectangles[index].pos[axis] = random.uniform(lower_limit, self.rectangles[index].pos[axis])
