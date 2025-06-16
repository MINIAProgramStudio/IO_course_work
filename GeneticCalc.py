import os

import numpy as np
import ImageContainer as IC
import fitness
from GeneticSolver import GeneticSolver

def genetic_calc(input_IC, pop_size, children, starting_overpop, dimensions, minmax, mutation_prob, mutation_pow, iterations, progressbar = False, stats = False, point = None):
    func = fitness.fitness_constructor(input_IC, [2,100,1,1,2], point)
    solver = GeneticSolver(func, pop_size, children, starting_overpop, dimensions, minmax, mutation_prob, mutation_pow, seeking_min =True, clever_mutation = False)
    if stats:
        result = solver.solve_stats(iterations, progressbar)
    else:
        result = solver.solve(iterations, progressbar, epsilon=0.5, epsilon_timeout=10)
    return result