from tqdm import tqdm

import ImageContainer as IC
from PSO import PSOSolver
from GeneticSolver import GeneticSolver
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import os



def IC_to_positions(input_IC: IC.ImageContainer):
    array = input_IC.array
    positions = np.argwhere(array[:, :, 1] > 0)
    p = np.zeros_like(positions)
    p[:, 0], p[:, 1] = positions[:, 1], positions[:, 0]
    return p

def order_quad_vertices(pos):

    points = np.round(np.array(pos).reshape((4, 2)))
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
    x, y = point # якогось біса треба міняти X та Y, я не знаю чому
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