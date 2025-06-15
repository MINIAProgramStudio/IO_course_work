import os

import numpy as np
import ImageContainer as IC
import fitness
from PSO import PSOSolver

def pso_calc(input_IC, coef, iterations, progressbar = False, stats = False, point = None):
    func = fitness.fitness_constructor(input_IC, [1,1,1,1,1], point)

    solver = PSOSolver(coef, func, seeking_min=True)
    if stats:
        result = solver.solve_stats(iterations, progressbar)
    else:
        result = solver.solve(iterations, progressbar)
    return result