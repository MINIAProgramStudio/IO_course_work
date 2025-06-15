import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import time
from pathos.multiprocessing import ProcessingPool as Pool
import os

class PSOParticle:
    def __init__(self, coef, func, seeking_min):
        self.c = coef
        self.pos = np.random.random_sample(self.c["dim"]) * (self.c["pos_max"]-self.c["pos_min"]) + self.c["pos_min"]
        self.speed =  np.random.random_sample(self.c["dim"]) * (self.c["speed_max"]-self.c["speed_min"]) + self.c["speed_min"]
        self.best_pos = self.pos
        self.func = func
        self.seeking_min = seeking_min
        self.best_val = self.func(self.pos)


    def update(self, global_best_pos, r1, r2):
        # оновлення швидкості
        self.speed = self.c["braking"] * self.speed + self.c["a1"] * (self.best_pos - self.pos) * r1 + self.c["a2"] * (global_best_pos - self.pos) * r2
        self.speed = np.minimum(self.c["speed_max"], np.maximum(self.c["speed_min"], self.speed))
        # оновлення позиції
        self.pos = self.pos + self.speed
        for d in range(self.c["dim"]):
            if self.pos[d] < self.c["pos_min"][d] or self.pos[d] > self.c["pos_max"][d]:
                self.pos[d] = np.random.uniform(self.c["pos_min"][d], self.c["pos_max"][d])
        # оновлення найкращої позиції
        result = self.func(self.pos)
        if ((result > self.best_val) and not self.seeking_min) or ((result < self.best_val) and self.seeking_min) :
            self.best_pos = self.pos
            self.best_val = result
            return (self.pos, result)
        else:
            return None

def update_particle(args):
    particle, global_best_pos, r1, r2 = args[0], args[1], args[2], args[3]
    r1 = np.random.random_sample(particle.c["dim"])
    r2 = np.random.random_sample(particle.c["dim"])
    return particle.update(global_best_pos, r1, r2)

"""
coef list:
"a1": ,# self acceleration number
"a2": ,# population acceleration number
"pop_size": ,#population size
"dim": ,#dimensions
"pos_min": ,#vector of minimum positions
"pos_max": ,#vector of maximum positions
"speed_min": ,#vector of min speed
"speed_max": ,#vector of max speed
"braking": ,#speed depletion
"""


class PSOSolver:
    def __init__(self, coef, func, seeking_min = False):
        self.c = coef
        self.func = func
        self.seeking_min = seeking_min
        self.pop = [PSOParticle(self.c, self.func, self.seeking_min) for i in range(self.c["pop_size"])]
        personal_best = [particle.best_val for particle in self.pop]
        if self.seeking_min:
            best_i = np.argmin(personal_best)
        else:
            best_i = np.argmax(personal_best)
        self.best_pos = self.pop[best_i].best_pos
        self.best_val = self.pop[best_i].best_val

    def reset(self):
        self.pop = [PSOParticle(self.c, self.func, self.seeking_min) for i in range(self.c["pop_size"])]
        personal_best = [particle.best_val for particle in self.pop]
        if self.seeking_min:
            best_i = np.argmin(personal_best)
        else:
            best_i = np.argmax(personal_best)
        self.best_pos = self.pop[best_i].best_pos
        self.best_val = self.pop[best_i].best_val

    def iter(self):
        new_best_val = self.best_val
        new_best_pos = self.best_pos
        r1 = np.random.random_sample(self.c["dim"])
        r2 = np.random.random_sample(self.c["dim"])
        iterables = [[p, self.best_pos, r1, r2] for p in self.pop]
        with Pool(os.cpu_count() * 2 // 3) as pool:
            results = pool.map(update_particle, iterables)
        for result in results:
            if result:
                if ((result[1] > self.best_val) and not self.seeking_min) or ((result[1] < self.best_val) and self.seeking_min):
                    new_best_val = result[1]
                    new_best_pos = result[0]
        positions = [p.pos for p in self.pop]
        for i in range(self.c["pop_size"]):
            for j in range(i + 1, self.c["pop_size"]):
                if np.array_equal(positions[i], positions[j]):
                    self.pop[i] = PSOParticle(self.c, self.func, self.seeking_min)
        self.best_val = new_best_val
        self.best_pos = new_best_pos

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc = "PSO")
        for iter in iterator:
            self.iter()
        return (self.best_val, self.best_pos)

    def anisolve(self, iterations = 100, save = False):
        if self.c["dim"] == 2:
            fig = plt.figure()
            ax = plt.axes(projection="3d", computed_zorder=False)
            dots_x = [particle.pos[0] for particle in self.pop]
            dots_y = [particle.pos[1] for particle in self.pop]
            dots_z = [self.func(particle.pos) for particle in self.pop]
            prime_x = [self.best_pos[0]]
            prime_y = [self.best_pos[1]]
            prime_z = [self.best_val]

            func_x = np.linspace(self.c["pos_min"][0], self.c["pos_max"][0], 100)
            func_y = np.linspace(self.c["pos_min"][1], self.c["pos_max"][1], 100)
            FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)

            FUNC_Z = self.func([FUNC_X, FUNC_Y])

            surface = ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                                      linewidth=0, antialiased=False, zorder=0)
            dots = ax.scatter(dots_x, dots_y, dots_z, c="#ff0000", zorder=5)
            prime = ax.scatter(prime_x, prime_y, prime_z, s=75, c="#ffff00", zorder=10)

            fig.suptitle("PSO " + str(0) + "/" + str(iterations) + " Best: " + str(round(self.best_val, 3)))

            def update(frame):
                self.iter()
                dots_x = [particle.pos[0] for particle in self.pop]
                dots_y = [particle.pos[1] for particle in self.pop]
                dots_z = [self.func(particle.pos) for particle in self.pop]
                prime_x = [self.best_pos[0]]
                prime_y = [self.best_pos[1]]
                prime_z = [self.best_val]

                dots.set_offsets(np.c_[dots_x, dots_y])
                dots.set_3d_properties(dots_z, zdir='z')
                prime.set_offsets(np.c_[prime_x, prime_y])
                prime.set_3d_properties(prime_z, zdir='z')
                fig.suptitle(
                    "PSO " + str(frame + 1) + "/" + str(iterations) + " Best: " + str(round(self.best_val, 3)))
                if frame >= iterations - 1:
                    ani.pause()
                return dots, prime

            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=50)
            if save:
                writervideo = animation.PillowWriter(fps=2, bitrate=1800)
                ani.save("gifs/pso_"+str(int(time()))+"_.gif", writer = writervideo)
                plt.show(block=False)
                plt.close("all")
            else:
                plt.show()
            return (self.best_val, self.best_pos)
        else:
            px = list(range(iterations + 1))
            py = [abs(self.best_val)]
            fig, ax = plt.subplots()
            for i in range(iterations):
                self.iter()
                py.append(abs(self.best_val))

            ax.set_yscale("log")

            graph = ax.plot(px, py)[0]
            if save:
                plt.savefig("gifs/pso_latest.png")
            plt.show()
            return (self.best_val, self.best_pos)

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []

        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="PSO")
        for _ in iterator:
            output.append(self.best_val)
            self.iter()
        output.append(self.best_val)
        return (self.best_val, self.best_pos, output)

    def solve_time(self, iterations = 100, progressbar = False):
        output = []

        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="PSO")
        start = time()
        for _ in iterator:
            output.append(time()-start)
            self.iter()
        output.append(time()-start)
        return (self.best_val, self.best_pos, output)
