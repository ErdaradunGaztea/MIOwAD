"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np

from genetic.GeneticLibrary import rn_gen_alg
from genetic.Stopper import Stopper

r3_gen_alg = rn_gen_alg(3, lambda genes: genes[0]**2 + genes[1]**2 + 2*genes[2]**2, size=250)\
    .set_stopper(Stopper().set_max_iter(100))
solution = r3_gen_alg.run()
print('\n')
print(solution.genes)

rastrigin_gen_alg = rn_gen_alg(
    5,
    lambda genes: 50 + np.sum(genes**2 - 10 * np.cos(2 * np.pi * genes)),
    size=250
).set_stopper(Stopper().set_max_iter(250))
solution = rastrigin_gen_alg.run()
print('\n')
print(solution.genes)
