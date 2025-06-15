import numpy as np
import ImageContainer as IC

def order_quad_vertices(pos):
    points = np.array(np.round(np.array(pos).reshape((4, 2))), dtype = np.int16)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    ordered = points[sort_idx]
    return ordered

class Quad:
    def __init__(self, pos):
        self.vertices = order_quad_vertices(pos)
        self.edge_lengths = [np.linalg.norm(self.vertices[i] - self.vertices[(i+1)%1])
                             for i in range(4)]
        self.edge_angles = [np.pi + np.arctan(
            (self.vertices[i][1]-self.vertices[(i+1)%4][1])/(self.vertices[i][0]-self.vertices[(i+1)%4][0] + 0.001)
        ) for i in range(4)]

        self.d1 = np.linalg.norm(self.vertices[0] - self.vertices[2])
        self.d2 = np.linalg.norm(self.vertices[1] - self.vertices[3])
        v1 = self.vertices[0] - self.vertices[2]
        v2 = self.vertices[1] - self.vertices[3]
        cos_theta = np.dot(v1, v2) / (self.d1 * self.d2 + 1e-12)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        self.space = 0.5 * self.d1 * self.d2 * sin_theta
        self.perimeter = sum(self.edge_lengths)


def length_equality(quad):
    return (quad.edge_lengths[0]-quad.edge_lengths[2])**2/(quad.edge_lengths[0]+quad.edge_lengths[2] +0.01) + \
        (quad.edge_lengths[1] - quad.edge_lengths[3]) ** 2 / (quad.edge_lengths[1] + quad.edge_lengths[3] +0.01)

def angle_equality(quad):
    return (quad.edge_angles[0] - quad.edge_angles[2]) ** 2 + \
        (quad.edge_angles[1] - quad.edge_angles[3]) ** 2

def bresenham(x0, y0, x1, y1):
    p = []
    delta_x = abs(x1 - x0)
    delta_y = abs(y1 - y0)
    sign_x = x1 >= x0
    sign_y = y1 >= y0

    if delta_x >= delta_y:
        d = x0
        s = y0
        delta_d = delta_x
        delta_s = delta_y
        sign_s = sign_y
        sign_d = sign_x
        d1 = x1
    else:
        d = y0
        s = x0
        delta_d = delta_y
        delta_s = delta_x
        sign_s = sign_x
        sign_d = sign_y
        d1 = y1
    error = np.floor(delta_d/2)
    while d < d1:
        if delta_x >= delta_y:
            p.append([d, s])
        else:
            p.append([s, d])
        error -= delta_s
        if error < 0:
            s += sign_s
            error += delta_s
        d += sign_d
    p.append([x1, y1])
    return np.array(p)

def bresenham_counter(x0, y0, x1, y1, input_IC):
    p = 0
    delta_x = abs(x1 - x0)
    delta_y = abs(y1 - y0)
    sign_x = x1 >= x0
    sign_y = y1 >= y0

    width = input_IC.array.shape[1] - 1
    height = input_IC.array.shape[0] - 1

    if delta_x >= delta_y:
        d = x0
        s = y0
        delta_d = delta_x
        delta_s = delta_y
        sign_s = sign_y
        sign_d = sign_x
        d1 = x1
        s1 = y1
    else:
        d = y0
        s = x0
        delta_d = delta_y
        delta_s = delta_x
        sign_s = sign_x
        sign_d = sign_y
        d1 = y1
        s1 = x1
    error = np.floor(delta_d/2)
    while d < d1:

        if delta_x >= delta_y:
            if d > width or s > height or d < 0 or s < 0:
                break
            p += input_IC.array[s][d][0]

        else:
            if s > width or d > height or d < 0 or s < 0:
                break
            p += input_IC.array[d][s][0]
        error -= delta_s
        if error < 0:
            s += sign_s
            error += delta_s
            if sign_s:
                s = min(s, s1-1)
        d += sign_d
    if 0 <= x1 and x1 <= width and 0 <= y1 and y1 <= height:
        p += input_IC.array[y1][x1][0]
    return p

def b_4(quad, input_IC):
    p_sum = 0
    for i in range(4):
        p_sum += bresenham_counter(quad.vertices[i][0], quad.vertices[i][1], quad.vertices[(i+1)%4][0], quad.vertices[(i+1)%4][1], input_IC)
    return max(p_sum, 0.00001)

def theta_b_4(quad, input_IC):
    return (quad.perimeter-b_4(quad, input_IC))**2

def is_point_in_quad(point, quad):
    x, y = point
    inside = False
    for i in range(4):
        xi, yi = quad.vertices[i]
        xj, yj = quad.vertices[(i + 1) % 4]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
    return inside

def theta_c(quad, input_IC_points):
    c = 0
    for point in input_IC_points:
        if is_point_in_quad(point, quad):
            c += 1
    return c+0.01

def theta_2d(quad):
    return (quad.d1-(quad.d2+quad.d1)/2)**2/(quad.d1 + 0.01) + (quad.d2-(quad.d2+quad.d1)/2)**2/(quad.d2 + 0.01)

def fitness_constructor(input_IC, consts, point = None):
    input_IC_points = IC.to_positions(input_IC)
    if point is None:
        def fitness(pos):
            quad = Quad(pos)
            return (length_equality(quad) + consts[0]) * \
                (angle_equality(quad) + consts[1]) * \
                (theta_b_4(quad, input_IC) + consts[2]) * \
                (theta_c(quad, input_IC_points) + consts[3]) * \
                (theta_2d(quad) + consts[4])
    else:
        def fitness(pos):
            quad = Quad(pos)
            isiq = is_point_in_quad(point, quad)
            if not isiq:
                isiq = float("inf")
            return isiq - theta_c(quad, input_IC_points) *consts[3]+ theta_b_4(quad, input_IC)*consts[2] + angle_equality(quad)*consts[1] + length_equality(quad)*consts[0] + theta_2d(quad)*consts[4] - quad.perimeter - quad.space
    return fitness


"""def fitness_constructor(input_IC, consts, point = None):
    input_IC_points = IC.to_positions(input_IC)
    if point is None:
        def fitness(pos):
            quad = Quad(pos)
            return (length_equality(quad) + consts[0]) * \
                (angle_equality(quad) + consts[1]) * \
                (theta_b_4(quad, input_IC) + consts[2]) * \
                (theta_c(quad, input_IC_points) + consts[3]) * \
                (theta_2d(quad) + consts[4])
    else:
        def fitness(pos):
            quad = Quad(pos)
            isiq = is_point_in_quad(point, quad)
            if not isiq:
                isiq = float("inf")
            return isiq + (theta_c(quad, input_IC_points) + consts[3]) * (theta_b_4(quad, input_IC) + consts[2])
            return (length_equality(quad) + consts[0]) * \
                (angle_equality(quad) + consts[1]) * \
                (theta_b_4(quad, input_IC) + consts[2]) * \
                (theta_c(quad, input_IC_points) + consts[3]) * \
                (theta_2d(quad) + consts[4]) + isiq
    return fitness"""