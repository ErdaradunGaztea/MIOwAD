"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np

from genetic.Crossover import Crossover
from genetic.GeneticAlgorithm import GeneticAlgorithm
from genetic.Mutation import Mutation
from genetic.Task import Task


def rn_gen_alg(dimensions, target, size=50, minimize=True):
    task = Task(target, lambda: np.random.normal(0, 5, (dimensions,)), minimize)
    return GeneticAlgorithm(task, size)\
        .set_crossover(
            Crossover(__get_crossover__(dimensions))
        ).set_mutation(
            Mutation(__get_mutation__(dimensions))
        )


def __get_crossover__(dimensions):
    """Crossover function generator."""
    def __crossover__(parents):
        split_point = np.random.randint(dimensions + 1)
        return np.concatenate((parents[0].genes[:split_point], parents[1].genes[split_point:]))
    return __crossover__


def __get_mutation__(dimensions):
    """Mutation function generator."""
    def __mutation__(indv):
        indv.genes += np.random.normal(0, 0.25, dimensions)
    return __mutation__
