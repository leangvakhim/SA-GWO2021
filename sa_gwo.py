import numpy as np
import math

class sa_gwo:
    def __init__(self, model, pop_size, max_iter, t_start, cooling_rate):
        self.model = model
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = model.num_sensors ** 2 # x and y for each sensor

        # sa parameters
        self.t = t_start
        self.cooling_rate = cooling_rate # me in eq 15

        # initialize population by random
        self.wolves = np.random.uniform(0, model.width, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)

        # initialize alpha, beta, delta
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = -float('inf') # maximize the coverage

        self.beta_pos = np.zeros(self.dim)
        self.beta_score = -float('inf')

        self.delta_pos = np.zeros(self.dim)
        self.delta_score = -float('inf')

    def optimize(self):
        convergence_curve = []

        for t in range(self.max_iter):
            # 1. calculate fitness and update alpha, beta and delta
            for i in range(self.pop_size):
                # clip the area
                self.wolves[i] = np.clip(self.wolves[i], 0, self.model.width)
                # calculate the fitness (coverage rate)
                self.fitness[i] = self.model.calculate_coverage(self.wolves[i])
                # update alpha (best), beta (2nd), delta (3rd)
                if self.fitness[i] > self.alpha_score:
                    self.alpha_score = self.fitness[i]
                    self.alpha_pos = self.wolves[i].copy()
                elif self.fitness[i] > self.beta_score:
                    self.beta_score = self.fitness[i]
                    self.beta_pos = self.wolves[i].copy()
                elif self.fitness[i] > self.delta_score:
                    self.delta_score = self.fitness[i]
                    self.delta_pos = self.wolves[i].copy()

            # 2. update gwo control parameter (a)
            a = 2 - t * (2 / self.max_iter)

            # 3. siege behavior (update postion of all wolves)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # update based on alpha
                    r1, r2 = np.random.random(), np.random.random()
                    # eq 8
                    A1 = 2 * a * r1 - a
                    # eq 9
                    C1 = 2 * r2
                    # eq 5
                    D_alpha = np.abs(C1 * self.alpha_pos[j] - self.wolves[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    # update based on beta
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    # eq 6
                    D_beta = np.abs(C2 * self.beta_pos[j] - self.wolves[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                     # update based on delta
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    # eq 7
                    D_delta = np.abs(C3 * self.delta_pos[j] - self.wolves[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.wolves[i, j] = (X1 + X2 + X3) / 3

            # 4. simulated annealing (SA) operation
            perturbation = (np.random.uniform(-1, 1, self.dim) * 5.0 * a)
            y_pos = self.alpha_pos + perturbation
            y_pos = np.clip(y_pos, 0, self.model.width)
            # calculate fitness of new solution y
            h_new = self.model.calculate_coverage(y_pos)
            h_current = self.alpha_score

            if h_new > h_current:
                # accept improved solution
                self.alpha_pos = y_pos
                self.alpha_score = h_new
            else:
                # accept bad solution with probability
                delta_h = h_current - h_new
                prob = math.exp(-(delta_h * 1000) / self.t)

                if np.random.random() < prob:
                    self.alpha_pos = y_pos
                    self.alpha_score = h_new

            # 5. cooling schedule
            self.t = self.t * self.cooling_rate

            convergence_curve.append(self.alpha_score)

        return self.alpha_pos, convergence_curve

