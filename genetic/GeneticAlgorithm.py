"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import random
import sys

import numpy as np

from genetic.Individual import Individual
from genetic.Selection import exponential_selection_min, exponential_selection_max
from genetic.Stopper import Stopper


class GeneticAlgorithm:
    def __init__(self, task, size):
        self.task = task
        self.selection = exponential_selection_min if task.minimize else exponential_selection_max
        self.crossover = None
        self.mutations = []
        # default stop condition is after 100 iterations
        self.stopper = Stopper().set_max_iter(100)
        self.solutions = np.empty((size,), Individual)

    def run(self):
        """Runs the algorithm until the stop condition is meet, then returns the best of solutions obtained."""
        self.initialize()
        while not self.stopper.stop(self):
            sys.stdout.write("Iteration no. {0}".format(self.stopper.iteration))
            sys.stdout.flush()
            self.cross()
            self.mutate()
            self.evaluate()
            self.select()
            sys.stdout.write("\rIteration no. {0}; best target: {1}\n".format(
                self.stopper.iteration, max(map(lambda x: x.target, self.solutions))))
        self.evaluate()
        return self.best_solution()

    def initialize(self):
        """Generates genes for each individual from array with preset size."""
        for index in range(len(self.solutions)):
            self.solutions[index] = Individual(self.task.generate_genes())

    def cross(self):
        new_solutions = np.empty_like(self.solutions)
        for index in range(len(new_solutions)):
            parents = random.sample(list(self.solutions), 2)
            new_solutions[index] = self.crossover.cross(parents)
        self.solutions = new_solutions

    def mutate(self):
        for mutation in self.mutations:
            for indv in self.solutions:
                mutation.mutate(indv)

    def evaluate(self):
        for indv in self.solutions:
            self.task.evaluate(indv)

    def select(self):
        self.solutions = self.selection.select(self.solutions)

    def best_solution(self):
        best_score = max(map(lambda x: x.score, self.solutions))
        return next(x for x in self.solutions if x.score == best_score)

    def set_crossover(self, crossover):
        self.crossover = crossover
        return self

    def set_mutation(self, mutation):
        self.mutations.append(mutation)
        return self

    def set_selection(self, selection):
        self.selection = selection
        return self

    def set_stopper(self, stopper):
        self.stopper = stopper
        return self
