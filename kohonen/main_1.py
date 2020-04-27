"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import pandas as pd

from kohonen.KohonenNet import KohonenNet
from kohonen.Neighbourhoods import neighbour_gauss, neighbour_mexican_hat
from kohonen.Task import Task
from kohonen.Topology import rectangle_eucl, rectangle

"""
Important note!
Whole project is based on the idea of interfaces. Each class that is really an interface is noted as such
in its description. Also, internal methods are marked as __internal_methods__ and shouldn't be called from outside.
"""

data = pd.read_csv('resources/kohonen/hexagon.csv')
target = 'c'
# Task class inspired by mlr (R package), useful when using multiple models for same task
task = Task(data, target)

# Usage examples:
"""
8x8 rectangular network where distances are computed using euclidean distance 
Radius of neighbourhood is set to 0.25, function to gauss and runs set to 10
"""
# used chaining to make code more readable
KohonenNet()\
    .add_task(task)\
    .add_topology(rectangle_eucl.instantiate([8, 8]))\
    .add_neighbourhood(neighbour_gauss.set_radius(0.25))\
    .train(runs=10)

"""
8x8 rectangular network where distances are computed using euclidean distance 
Radius of neighbourhood is set to 0.1, function to "mexican hat" and runs set to 10
"""
# train also returns self to allow further whatever
kn = KohonenNet()\
    .add_task(task)\
    .add_topology(rectangle.instantiate([3, 3]))\
    .add_neighbourhood(neighbour_mexican_hat.set_radius(0.1))\
    .train(runs=10)
