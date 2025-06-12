from tqdm import tqdm

import ImageContainer as IC
from PSO import PSOSolver
from GeneticSolver import GeneticSolver
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import os

def IC_to_positions(input_IC: IC.ImageContainer):
    array = input_IC.array
    positions = np.argwhere(array[:, :, 1] == 1)
    return positions

def order_quad_vertices(pos):

    pos = np.array(pos).reshape((4, 2))
    points = np.zeros_like(pos)
    points[:, 0], points[:, 1] = pos[:, 1], pos[:, 0]
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    ordered = points[sort_idx]

    return ordered

def point_to_segment_distance(p, a, b):
    # p is (2,), a and b are (2,)
    ap = p - a
    ab = b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def distance_for_point_constructor(length_func, edges):
    def distance_for_point(position):
        return min([length_func(position, a, b) for a, b in edges])
    return distance_for_point

def is_point_in_polygon(point, polygon):
    # polygon - масив вершин (N, 2), point - (2,)
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
    return inside

def distance_complex(positions, length_func, polygon):
    edges = [(polygon[i], polygon[(i + 1) % 4]) for i in range(4)]
    distance_for_point = distance_for_point_constructor(length_func, edges)
    dists = np.array(list(map(distance_for_point, positions)))
    mean_dists = np.mean(dists)
    for i in range(positions.shape[0]):
        if is_point_in_polygon(positions[i], polygon):
            dists[i] = -((dists[i]*100)**8)
    dists[dists > mean_dists//20] = 0
    dists[dists>0] = mean_dists/(dists[dists>0]+0.01) # more value in lower distances
    return positions.shape[0]/sum(dists)*(np.sum(dists>0))

def distance_lined(positions, length_func, polygon):
    edges = [(polygon[i], polygon[(i + 1) % 4]) for i in range(4)]
    perimeter = np.sum([np.sqrt(np.sum((polygon[i]-polygon[(i+1)% 4])**2)) for i in range(4)])
    distance_for_point = distance_for_point_constructor(length_func, edges)
    dists = np.array(list(map(distance_for_point, positions)))
    return ((max(perimeter, 80)-np.sum(dists < 1))**2)/perimeter + 0.000001

def distance_lined_complex(positions, length_func, polygon):
    edges = [(polygon[i], polygon[(i + 1) % 4]) for i in range(4)]
    perimeter = np.sum([np.sqrt(np.sum((polygon[i]-polygon[(i+1)% 4])**2)) for i in range(4)])
    distance_for_point = distance_for_point_constructor(length_func, edges)
    dists = np.array(list(map(distance_for_point, positions)))
    for i in range(positions.shape[0]):
        if is_point_in_polygon(positions[i], polygon):
            dists[i] = -dists[i]
    return ((max(perimeter, 80)-np.sum(dists < 1))**2)/perimeter + (np.abs(np.sum(dists[dists<-1]))+1)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def is_convex_quad(polygon):
    def vec(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    n = len(polygon)
    signs = []
    for i in range(n):
        a = vec(polygon[i], polygon[(i+1) % n])
        b = vec(polygon[(i+1) % n], polygon[(i+2) % n])
        cp = cross(a, b)
        signs.append(cp > 0)

    return all(signs) or not any(signs)


def rectangleness(polygon):
    edge_lengths = [np.sqrt(np.sum((polygon[i] - polygon[(i + 1) % 4]) ** 2))
                    for i in range(4)]
    mean_length = np.mean(edge_lengths)
    MSE_length = np.sum([(1-edge/mean_length)**2
                         for edge in edge_lengths])

    diag_lengths = [np.sqrt(np.sum((polygon[i] - polygon[(i + 2) % 4]) ** 2))
                    for i in range(2)]
    mean_diag_lengths = np.mean(diag_lengths)
    MSE_diag = np.sum([(1-diag/mean_diag_lengths)**2
                         for diag in diag_lengths])
    if is_convex_quad(polygon):
        return (max(MSE_length,MSE_diag)+0.1)
    else:
        return float("inf")

def fitness_constructor(positions, length_func, rect_func):
    def fitness(pos):
        if np.any(np.array(pos)<0) or np.any(np.array(pos) > np.max(positions)+5):
            return float("inf")
        polygon = order_quad_vertices(pos)
        dist = distance_lined_complex(positions, length_func, polygon)
        rect = rect_func(polygon)
        return dist*rect
    return fitness

def PSOBed_finder(input_IC, progressbar = False, stats = False):
    temp_IC = IC.average_channels(input_IC)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    temp_IC = IC.apply_kernel(temp_IC, kernel)
    temp_IC = IC.dynamic_binarize(temp_IC, 0.075, 0.001, 0.001)
    positions = IC_to_positions(temp_IC)
    fit_func = fitness_constructor(positions, point_to_segment_distance, rectangleness)
    solver = PSOSolver({
        "a1": 1.5,  # self acceleration number
        "a2": 2,  # population acceleration number
        "pop_size": 30,  # population size
        "dim": 8,  # dimensions
        "pos_min": np.zeros(8),  # vector of minimum positions
        "pos_max": np.array([temp_IC.array.shape[0], temp_IC.array.shape[1]]*4),  # vector of maximum positions
        "speed_min": np.ones(8)*(-np.sqrt(temp_IC.array.shape[0]**2+temp_IC.array.shape[1]**2)/5),  # vector of min speed
        "speed_max": np.ones(8)*(np.sqrt(temp_IC.array.shape[0]**2+temp_IC.array.shape[1]**2)/5),  # vector of max speed
        "braking": 0.7
    }, fit_func, seeking_min=True)
    if stats:
        result = solver.solve(100, progressbar = progressbar)
    else:
        result = solver.solve_stats(100, progressbar=progressbar)
    result = list(result)
    result[1] = order_quad_vertices(result[1])
    return result

def Genetic_bed_finder(input_IC, progressbar = False, stats = False):
    temp_IC = IC.average_channels(input_IC)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    temp_IC = IC.apply_kernel(temp_IC, kernel)
    temp_IC = IC.dynamic_binarize(temp_IC, 0.075, 0.001, 0.001)
    positions = IC_to_positions(temp_IC)
    fit_func = fitness_constructor(positions, point_to_segment_distance, rectangleness)
    solver = GeneticSolver(fit_func, 100, 400, 2000, 8, np.array([[0,temp_IC.array.shape[0]], [0,temp_IC.array.shape[1]]]*4), 0.05, 0.2, seeking_min=True)
    if stats:
        result = solver.solve(100, progressbar=progressbar)
    else:
        result = solver.solve_stats(100, progressbar=progressbar, epsilon=10**(-2), epsilon_timeout=5)
    result = list(result)
    result[1] = order_quad_vertices(result[1])
    return result

def Random_bed_finder(input_IC, iterations = 10**4, progressbar = False, stats = False):
    temp_IC = IC.average_channels(input_IC)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    temp_IC = IC.apply_kernel(temp_IC, kernel)
    temp_IC = IC.dynamic_binarize(temp_IC, 0.075, 0.001, 0.001)
    positions = IC_to_positions(temp_IC)
    fit_func = fitness_constructor(positions, point_to_segment_distance, rectangleness)

    positions = np.random.rand(iterations, 4, 2) * [temp_IC.array.shape[0], temp_IC.array.shape[1]]

    if progressbar:
        with Pool(os.cpu_count() * 2 // 3) as pool:
            results = list(tqdm(pool.imap(fit_func, positions), total=len(positions)))
    else:
        with Pool(os.cpu_count() * 2 // 3) as pool:
            results = list(pool.imap(fit_func, positions))
    results = np.array(results)
    best_id = np.argmin(results)
    return (results[best_id], positions[best_id])
