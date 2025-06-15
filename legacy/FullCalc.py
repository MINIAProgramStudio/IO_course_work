import os

import numpy as np
import ImageContainer as IC
import fitness
from tqdm import tqdm
import math
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import combinations

def fullcalc(input_IC, point = None):
    func = fitness.fitness_constructor(input_IC, [1,1,1,1, 1], point)
    input_points = IC.to_positions(input_IC)
    N = input_points.shape[0]
    iterations = math.factorial(N) // (math.factorial(4) * math.factorial(N - 4))
    iterator = tqdm(total=iterations, desc = "Повний перебір")
    best = [input_points[0][0], input_points[0][1],
            input_points[1][0], input_points[1][1],
            input_points[2][0], input_points[2][1],
            input_points[3][0], input_points[3][1]]
    best_value = func(best)
    for i_0 in range(N):
        for i_1 in range(i_0+1, N):
            for i_2 in range(i_1+1, N):
                for i_3 in range(i_2+1, N):
                    pos = [input_points[i_0][0], input_points[i_0][1],
                        input_points[i_1][0], input_points[i_1][1],
                        input_points[i_2][0], input_points[i_2][1],
                        input_points[i_3][0], input_points[i_3][1]]
                    value = func(pos)
                    if value < best_value:
                        best = pos
                    iterator.update(1)
    return (best_value, best)

def fullcalc_paralel(input_IC):
    func = fitness.fitness_constructor(input_IC, [1, 1, 1, 1])
    input_points = IC.to_positions(input_IC)
    N = input_points.shape[0]
    iterations = math.factorial(N) // (math.factorial(4) * math.factorial(N - 4))
    iterator = tqdm(total=iterations, desc="Повний перебір")
    best = [input_points[0][0], input_points[0][1],
            input_points[1][0], input_points[1][1],
            input_points[2][0], input_points[2][1],
            input_points[3][0], input_points[3][1]]
    best_value = func(best)
    index_combinations = list(combinations(range(N), 4))

    def process_comb(indices):
        i_0, i_1, i_2, i_3 = indices
        pos = [
            input_points[i_0][0], input_points[i_0][1],
            input_points[i_1][0], input_points[i_1][1],
            input_points[i_2][0], input_points[i_2][1],
            input_points[i_3][0], input_points[i_3][1]
        ]
        value = func(pos)
        return value, pos

    best_value = float('inf')
    best = None
    with Pool(2) as pool:
        for value, pos in tqdm(pool.imap(process_comb, index_combinations), total=len(index_combinations), desc = "Паралельний повний перебір"):
            if value < best_value:
                best_value = value
                best = pos
    return (best_value, best)