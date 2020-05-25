"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import random


class Mutation:
    def __init__(self, mutate_func):
        # a function that takes one argument, an individual
        self.mutate_func = mutate_func
        # default algorithm parameters
        self.probability = 0.2

    def set_probability(self, probability):
        self.probability = probability
        return self

    def mutate(self, indv):
        if random.random() < self.probability:
            self.mutate_func(indv)
