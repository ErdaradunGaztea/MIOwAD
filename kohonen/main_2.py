"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from kohonen.KohonenNet import KohonenNet
from kohonen.Neighbourhoods import neighbour_gauss
from kohonen.Task import Task
from kohonen.Topology import rectangle, hexagonal_rectangle

"""
Important note!
Whole project is based on the idea of interfaces. Each class that is really an interface is noted as such
in its description. Also, internal methods are marked as __internal_methods__ and shouldn't be called from outside.
"""

# this is perfect as data doesn't need to be supplied through Github
mnist = fetch_openml('mnist_784')
# necessary steps to prepare data for Task class
data = pd.DataFrame(
    np.hstack((mnist.data, mnist.target.reshape((mnist.target.shape[0], 1)))),
    columns=mnist.feature_names + mnist.target_names,
    dtype=np.int64
)
# Task class inspired by mlr (R package), useful when using multiple models for same task
task = Task(data, mnist.target_names[0]).generate_pca()

kn = KohonenNet()\
    .add_task(task)\
    .add_topology(hexagonal_rectangle.instantiate([4, 4]))\
    .add_neighbourhood(neighbour_gauss.set_radius(0.25))\
    .train(runs=25)
