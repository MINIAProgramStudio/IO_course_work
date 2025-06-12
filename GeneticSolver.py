from random import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import os

class GeneticSolver:
    def __init__(self, func, pop_size, children, starting_overpop, dimensions, minmax, mutation_prob, mutation_pow, seeking_min = False):
        # запам'ятовуємо гіперпараметри
        self.func = func
        self.pop_size = pop_size
        self.children = children
        self.dimensions = dimensions
        self.minmax = minmax
        self.mutation_prob = mutation_prob
        self.mutation_pow = mutation_pow
        self.seeking_min = seeking_min

        #створюємо хромосоми
        self.pop = [[random()*(minmax[d][1] - minmax[d][0]) + minmax[d][0] for d in range(dimensions)] for _ in range(pop_size)]

    def reset(self):
        self.pop = [[random() * (self.minmax[d][1] - self.minmax[d][0]) + self.minmax[d][0] for d in range(self.dimensions)] for _ in range(self.pop_size + self.children)]

    def select(self):
        with Pool(os.cpu_count()*2//3) as pool:
            # Паралельно обчислити значення func для кожної хромосоми
            fitness_values = pool.map(self.func, self.pop)
        pop_with_fitness = list(zip(self.pop, fitness_values))
        pop_with_fitness.sort(key=lambda x: x[1], reverse=not self.seeking_min)
        self.pop = [chrom for chrom, fit in pop_with_fitness[:self.pop_size]]
        return self.pop[0]

    def crossover(self):
        for _ in range(self.children): # зробити children нових хромосом
            # псевдовипадкова генерація індексів батьків
            a = int(random()*self.pop_size)
            b = int(random()*(self.pop_size-1))
            if b >= a: b += 1
            # створити хромосому та додати її до популяції
            self.pop.append([
                random()*abs(self.pop[a][d] - self.pop[b][d]) + min(self.pop[a][d], self.pop[b][d]) for d in range(self.dimensions)
            ])

    def mutate(self):
        for i in range(1, len(self.pop)): # для кожної хромосоми окрім найкращої
            for d in range(self.dimensions): # для кожної координати (для кожного гена)
                if random() < self.mutation_prob: # з імовірністю mutation_prob
                    self.pop[i][d] = self.pop[i][d] + ((-1)**(int(random()*2))) * self.mutation_pow * random() * (self.minmax[d][1] - self.minmax[d][0]) #
                    self.pop[i][d] = min(self.minmax[d][1], max(self.minmax[d][0], self.pop[i][d]))

    def mutate_clever(self):
        for i in range(self.pop_size//10, len(self.pop)):  # для кожної хромосоми окрім найкращої
            if random() < self.mutation_prob:  # з імовірністю mutation_prob
                average_pos = [np.sum(self.pop[i][::2]), np.sum(self.pop[i][1::2])]
                move_x = average_pos[0]*(1 + random()*self.mutation_pow - self.mutation_pow)
                move_y = average_pos[1] * (1 + random() * self.mutation_pow - self.mutation_pow)
                self.pop[i][::2] += move_x
                self.pop[i][1::2] += move_y

            if random() < self.mutation_prob:  # з імовірністю mutation_prob
                average_pos = [np.sum(self.pop[i][::2]), np.sum(self.pop[i][1::2])]
                pos = np.array(self.pop[i])
                pos[::2] -= average_pos[0]
                pos[1::2] -= average_pos[1]
                rotate = random()*self.mutation_pow - self.mutation_pow
                pos[::2] = pos[::2] * np.cos(rotate) - pos[1::2] * np.sin(rotate)
                pos[1::2] = pos[::2] * np.sin(rotate) + pos[1::2] * np.cos(rotate)
                pos[::2] += average_pos[0]
                pos[1::2] += average_pos[1]
                self.pop[i] = list(pos)
            for d in range(self.dimensions):
                if random() < self.mutation_prob/8:
                    self.pop[i][d] = self.pop[i][d] + ((-1)**(int(random()*2))) * self.mutation_pow * random() * (self.minmax[d][1] - self.minmax[d][0]) #
                    self.pop[i][d] = min(self.minmax[d][1], max(self.minmax[d][0], self.pop[i][d]))


    def iter(self):
        self.crossover()
        self.mutate_clever()
        self.select()

    def solve(self, iterations, progressbar = True, epsilon_timeout = float("inf"), epsilon = 0):
        if progressbar:
            iterator = tqdm(range(iterations))
        else:
            iterator = range(iterations)
        epsilon_timeout_counter = 0

        for _ in iterator:
            old_best = self.pop[0]
            self.iter()

            if self.func(self.pop[0]) - self.func(old_best) >= epsilon and not self.seeking_min:
                epsilon_timeout_counter = 0
            elif self.func(old_best) - self.func(self.pop[0]) >= epsilon and self.seeking_min:
                epsilon_timeout_counter = 0
            else:
                epsilon_timeout_counter += 1
                if epsilon_timeout_counter > epsilon_timeout:
                    break
        self.select()
        return (self.func(self.pop[0]), self.pop[0])

    def solve_stats(self, iterations, progressbar = True, epsilon_timeout = float("inf"), epsilon = 0, show = False):
        y = []
        if progressbar:
            iterator = tqdm(range(iterations))
        else:
            iterator = range(iterations)
        epsilon_timeout_counter = 0

        for _ in iterator:
            old_best = self.pop[0]

            self.crossover()
            self.mutate_clever()
            self.select()
            y.append(self.func(self.pop[0]))

            if self.func(self.pop[0]) - self.func(old_best) >= epsilon and not self.seeking_min:
                epsilon_timeout_counter = 0
            elif self.func(old_best) - self.func(self.pop[0]) >= epsilon and self.seeking_min:
                epsilon_timeout_counter = 0
            else:
                epsilon_timeout_counter += 1
                if epsilon_timeout_counter > epsilon_timeout:
                    break
        self.select()
        y.append(self.func(self.pop[0]))
        if show:
            x = range(len(y))
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(x[:],y[:])
            # plt.yscale("log")
            ax.set_xlabel("iteration")
            ax.set_ylabel("best value")
            plt.show()
        return (self.func(self.pop[0]), self.pop[0], y)

    def anisolve(self, iterations, save = False):
        if self.dimensions != 2:
            raise Exception("GeneticSolver.anisolve can visualise only 2d functions")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d",computed_zorder=False)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        func_x = np.linspace(self.minmax[0][0], self.minmax[0][1], 106)
        func_y = np.linspace(self.minmax[1][0], self.minmax[1][1], 106)
        FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)
        FUNC_Z = self.func([FUNC_X, FUNC_Y])

        ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, zorder = 0)
        dots = ax.scatter([], [], [], c="#ff0000", zorder=5, label="Population")
        prime = ax.scatter([], [], [], s=75, c="#ffff00", zorder=10, label="Best Individual")

        ax.legend()
        ax.grid(True)

        def update(frame):
            self.select()
            fig.suptitle("Genetic" + str(frame + 1) + "/" + str(iterations) + " Best: " + str(
                round(self.func(self.pop[0]), 12)))
            x_coords = [p[0] for p in self.pop]
            y_coords = [p[1] for p in self.pop]
            z_coords = [self.func(p) for p in self.pop]

            dots._offsets3d = (x_coords, y_coords, z_coords)
            prime._offsets3d = ([self.pop[0][0]],
                                [self.pop[0][1]],
                                [self.func(self.pop[0])])
            self.crossover()
            self.mutate()
            if frame >= iterations - 1:
                ani.pause()
            return dots, prime
        if save:
            writervideo = animation.PillowWriter(fps=5, bitrate=1800)
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=100)
            ani.save("gifs/genetic_latest.gif", writer = writervideo)
        else:
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=100)
            plt.show()