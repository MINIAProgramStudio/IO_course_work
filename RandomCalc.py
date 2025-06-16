import os

import numpy as np
import ImageContainer as IC
import fitness
from tqdm import tqdm
import math
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import combinations

def random_calc_parallel(input_IC, iterations, point = None):
    func = fitness.fitness_constructor(input_IC, [1, 1, 1, 1, 1], point)
    width = input_IC.array.shape[1]
    height = input_IC.array.shape[0]
    def process_comb(pos):
        value = func(pos)
        return value, pos

    best_value = float('inf')
    best = None
    y = []
    positions = np.random.randint(np.zeros(8),[width, height, width, height, width, height, width, height], size = (iterations, 8)).tolist()
    with Pool(os.cpu_count()//3) as pool:
        counter = 0
        for value, pos in tqdm(pool.imap(process_comb, positions), total=len(positions), desc = "Паралельний випадковий перебір"):
            if value < best_value:
                best_value = value
                y.append([counter, best_value])
                best = pos
            counter += 1
    return (best_value, best, y)