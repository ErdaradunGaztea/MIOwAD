"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""
import copy

import numpy as np
import pandas as pd

from genetic.Crossover import Crossover
from genetic.GeneticAlgorithm import GeneticAlgorithm
from genetic.Mutation import Mutation
from genetic.Selection import Selection
from genetic.Stopper import Stopper
from genetic.Task import Task
from genetic.cutting_stock.Circle import Circle
from genetic.cutting_stock.Rectangle import Rectangle
from genetic.cutting_stock.RectangleTemplate import RectangleTemplate


def __get_circle_generator__(radius):
    # read circle data
    data = pd.read_csv("resources/cutting/r{0}.csv".format(radius), names=['width', 'height', 'value'])

    def __place_rect_in_circle__():
        # select random rectangle template from data
        row = data.sample(n=1).iloc[0]
        rect_template = RectangleTemplate([row.get('width'), row.get('height')], row.get('value'))
        # first create and rotate the rectangle randomly
        rect = Rectangle(rect_template, [0, 0])
        np.random.shuffle(rect.size)
        # then randomize its placement
        minus_half_width = -rect.size[0] / 2
        minus_half_height = -rect.size[1] / 2
        left_limit = -np.sqrt(radius ** 2 - minus_half_height ** 2)
        right_limit = 2 * minus_half_width - left_limit
        pos_x = np.random.uniform(left_limit, right_limit)
        bottom_limit = minus_half_height - np.sqrt(radius ** 2 - ((minus_half_width - abs(pos_x - minus_half_width)) ** 2 + minus_half_height ** 2))
        top_limit = 2 * minus_half_height - bottom_limit
        pos_y = np.random.uniform(bottom_limit, top_limit)
        rect.pos = np.array([pos_x, pos_y])
        # finally create circle with this rectangle inserted
        circle = Circle(radius)
        return circle.insert_rectangle(rect)
    return __place_rect_in_circle__


def __crossover__(parents):
    circle = copy.deepcopy(parents[0].genes)
    rectangles = np.random.choice(parents[1].genes.rectangles, int(np.ceil(len(parents[1].genes.rectangles) / 4)))
    for rect in rectangles:
        circle.insert_rectangle(rect)
    return circle


r = 800
task = Task(
    lambda genes: sum(map(lambda rect: rect.value, genes.rectangles)),
    __get_circle_generator__(r),
    minimize=False
)
gen_alg = GeneticAlgorithm(task, 250)\
    .set_crossover(Crossover(__crossover__))\
    .set_mutation(Mutation(lambda indv: indv.genes.flip(0)).set_probability(0.2))\
    .set_mutation(Mutation(lambda indv: indv.genes.flip(1)).set_probability(0.2))\
    .set_mutation(Mutation(lambda indv: indv.genes.random_push(0)).set_probability(0.6))\
    .set_mutation(Mutation(lambda indv: indv.genes.random_push(1)).set_probability(0.6))\
    .set_selection(Selection(lambda indv, solutions: np.exp(indv.target / 2500)))\
    .set_stopper(Stopper().set_max_iter(100))
