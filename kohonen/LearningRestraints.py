"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np


class LearningRestraint:
    def __init__(self, restraint_func):
        """Interface. Computes value used as multiplier to decay learning with time. Should return value from
        [0, 1] range."""
        self.__restrain__ = restraint_func
        self.params = {}

    def set_params(self, params):
        """Allows setting parameters in form of a dictionary."""
        self.params = params
        return self

    def update_params(self, params):
        """Allows updating parameters in the dictionary, keeping previously set parameters."""
        self.params.update(params)
        return self

    def restrain(self, t):
        """Computes restrain value depending on t (and parameters set before training)."""
        return self.__restrain__(t, self.params)


learning_exp = LearningRestraint(
    lambda t, params: np.exp(-t / params["coef"])
)
