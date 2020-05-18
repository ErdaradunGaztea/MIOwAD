"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import copy
import random

import numpy as np


class Selection:
    """Roulette selection, at least kind of."""
    def __init__(self, score_func):
        self.score = score_func
        self.probability = lambda indv, sol: indv.score / sum(map(lambda x: x.score, sol))

    def select(self, solutions):
        # compute score (based on target)
        for indv in solutions:
            indv.score = self.score(indv, solutions)
        # compute probability of being chosen (based on scores)
        probabilities = np.cumsum([self.probability(indv, solutions) for indv in solutions])
        # choose individuals with probability of being chosen once equal to its probability
        new_solutions = np.empty_like(solutions)
        for index in range(len(new_solutions)):
            # choose index of first element that's greater than random value
            indv_index = next(i for i, p in enumerate(probabilities) if p >= random.random())
            # COPY chosen element into new_solutions list
            new_solutions[index] = copy.deepcopy(solutions[indv_index])
        return new_solutions


# maximalization selector
exponential_selection_max = Selection(
    lambda indv, solutions: np.exp(indv.target)
)
# minimalization selector
exponential_selection_min = Selection(
    lambda indv, solutions: np.exp(-indv.target)
)
