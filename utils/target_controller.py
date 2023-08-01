import numpy as np
import math
from utils.tsp_controller import TSPSolver


class VTSPGaussian:
    def __init__(self, n_targets=2):
        self.n_targets = n_targets
        self.n_tsp_nodes = 50
        self.tsp_coord = self.get_tsp_nodes()
        self.tsp_idx = [0] * self.n_targets
        self.mean = self.tsp_coord[:, 0, :]
        self.sigma = np.array([0.1] * self.n_targets)
        self.max_value = 1 / (2 * np.pi * self.sigma ** 2)
        self.trajectories = [self.mean.copy()]

    def get_tsp_nodes(self):
        tsp_solver = TSPSolver()
        coord = np.random.rand(self.n_targets, self.n_tsp_nodes, 2)
        for i in range(self.n_targets):
            index = tsp_solver.run_solver(coord[i])
            coord[i] = coord[i][index]
        return coord

    def step(self, steplen):
        for i in range(self.n_targets):
            d = np.linalg.norm(self.tsp_coord[i, self.tsp_idx[i]+1, :] - self.mean[i])
            next_len = steplen
            if d > next_len:
                pt = (self.tsp_coord[i, self.tsp_idx[i]+1, :] - self.mean[i]) * next_len / d + self.mean[i]
            else:
                while True:
                    self.tsp_idx[i] += 1
                    next_len -= d
                    d = np.linalg.norm(self.tsp_coord[i, self.tsp_idx[i]+1, :] - self.tsp_coord[i, self.tsp_idx[i], :])
                    if d > next_len:
                        pt = (self.tsp_coord[i, self.tsp_idx[i]+1, :] - self.tsp_coord[i, self.tsp_idx[i], :]) * next_len / d + self.mean[i]
                        break
            self.mean[i] = pt
        self.trajectories += [self.mean.copy()]
        return self.mean

    def fn(self, X):
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            gaussian_mean = self.mean[target_id]
            sigma_x1 = sigma_x2 = self.sigma[target_id]
            covariance = 0
            r = covariance / (sigma_x1 * sigma_x2)
            coefficients = 1 / (2 * math.pi * sigma_x1 * sigma_x2 * np.sqrt(1 - math.pow(r, 2)))
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power((row_mat - gaussian_mean[0]) / sigma_x1, 2)
            py = np.power((col_mat - gaussian_mean[1]) / sigma_x2, 2)
            pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (sigma_x1 * sigma_x2)
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y[:, target_id] += distribution_matrix
        y /= self.max_value
        return y



if __name__ == '__main__':
    pass
