"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

from genetic.Individual import Individual


class Crossover:
    def __init__(self, cross_func):
        self.cross_func = cross_func

    def cross(self, parents):
        genes = self.cross_func(parents)
        return Individual(genes)
