import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product, combinations
from utils.tsp_controller import TSPSolver


class Gaussian2D:
    def __init__(self):
        self.rangeDistribs = (8,12)
        self.numDistribs = np.random.randint(self.rangeDistribs[0], self.rangeDistribs[1] + 1)
        self.mean = []
        self.sigma_x = []
        self.sigma_y = []
        self.cov = []
        self.max_value = None
        self.createProbability()

    def createProbability(self):
        def addGaussian():
            gaussian_mean = np.random.rand(2)
            gaussian_var = np.zeros((2, 2))
            gaussian_var[([0, 1], [0, 1])] = np.random.uniform(0.00005, 0.0002, 2)
            SigmaX = np.sqrt(gaussian_var[0][0])
            SigmaY = np.sqrt(gaussian_var[1][1])
            Covariance = gaussian_var[0][1]
            self.mean.append(gaussian_mean)
            self.sigma_x.append(SigmaX)
            self.sigma_y.append(SigmaY)
            self.cov.append(Covariance)
        for _ in range(self.numDistribs):
            addGaussian()

    def fn(self, X):
        y = np.zeros(X.shape[0])
        row_mat, col_mat = X[:,0], X[:,1]
        for gaussian_mean, SigmaX1, SigmaX2, Covariance in zip(self.mean, self.sigma_x, self.sigma_y, self.cov):
            r = Covariance / (SigmaX1 * SigmaX2)
            coefficients = 1 / (2 * math.pi * SigmaX1 * SigmaX2 * np.sqrt(1 - math.pow(r, 2)))
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power(row_mat - gaussian_mean[0], 2) / SigmaX1
            py = np.power(col_mat - gaussian_mean[1], 2) / SigmaX2
            pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (SigmaX1 * SigmaX2)
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y += distribution_matrix
        # y /= np.max(y)
        # print(y)
        if self.max_value is None:
            # print(y.shape)
            #assert y.shape == (2500,)
            self.max_value = np.max(y)
            y /= self.max_value
        else:
            y /= self.max_value
        return y

    @staticmethod
    def plot(img, i=0):
        plt.imshow(img)
        plt.title(i)
        plt.colorbar()
        plt.show()


class DriftGaussian:
    def __init__(self, n_targets=2, steplen=0.01, n_steps=150):
        self.n_targets = n_targets
        self.n_gaussians = np.random.randint(8, 13)
        self.max_steplen = steplen
        self.max_varsteplen = steplen / 5
        self.shape = (self.n_targets, 1, self.n_gaussians, 2)
        self.mean = []  # (n_targets, n_steps, n_gaussians, 2)
        self.sigma = []  # (n_targets, n_steps, n_gaussians, 2)
        self.mean_vel = []  # (n_targets, n_steps, n_gaussians, 2)
        self.sigma_vel = []  # (n_targets, n_steps, n_gaussians, 2)
        self.step = 0
        self.n_steps = n_steps + 2
        self.create_prob()
        self.run_drift_model()
        self.max_value = None

    def create_prob(self):
        self.mean = np.random.rand(*self.shape)
        self.sigma = np.random.uniform(0.05, 0.2, self.shape)
        self.mean_vel = np.random.uniform(-self.max_steplen, self.max_steplen, self.shape)
        self.sigma_vel = np.random.uniform(-self.max_varsteplen, self.max_varsteplen, self.shape)

    def run_drift_model(self):
        for _ in range(self.n_steps-1):
            delta_mean_vel = np.random.uniform(-self.max_steplen/2, self.max_steplen/2, self.shape)
            delta_sigma_vel = np.random.uniform(-self.max_varsteplen/2, self.max_varsteplen/2, self.shape)
            new_mean_vel = np.clip(delta_mean_vel + self.mean_vel[:, -1:, ...], -self.max_steplen, self.max_steplen)
            new_sigma_vel = np.clip(delta_sigma_vel + self.sigma_vel[:, -1:, ...], -self.max_varsteplen, self.max_varsteplen)
            new_mean = np.clip(self.mean[:, -1:, ...] + new_mean_vel, 0, 1)
            new_sigma = np.clip(self.sigma[:, -1:, ...] + new_sigma_vel, 0.05, 0.2)
            self.mean = np.concatenate((self.mean, new_mean), axis=1)
            self.sigma = np.concatenate((self.sigma, new_sigma), axis=1)
            self.mean_vel = np.concatenate((self.mean_vel, new_mean_vel), axis=1)
            self.sigma_vel = np.concatenate((self.sigma_vel, new_sigma_vel), axis=1)

    def fn(self, X, update=False):
        if update:
            self.step += 1
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            for gaussian_id in range(self.n_gaussians):
                gaussian_mean = self.mean[target_id][self.step][gaussian_id]
                sigma_x1, sigma_x2 = self.sigma[target_id][self.step][gaussian_id]
                covariance = 0
                r = covariance / (sigma_x1 * sigma_x2)
                coefficients = 1 / (2 * math.pi * sigma_x1 * sigma_x2 * np.sqrt(1 - math.pow(r, 2)))
                p1 = -1 / (2 * (1 - math.pow(r, 2)))
                px = np.power((row_mat - gaussian_mean[0]) / sigma_x1, 2)
                py = np.power((col_mat - gaussian_mean[1]) / sigma_x2, 2)
                pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (sigma_x1 * sigma_x2)
                distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
                y[:, target_id] += distribution_matrix
        if self.max_value is None:
            self.max_value = y.max(0) * 1.5  # estimate max
        y /= self.max_value
        return y


class TSPGaussian:
    def __init__(self, n_targets=2, steplen=0.01, n_steps=150):
        self.n_tsp_nodes = 100
        self.n_targets = n_targets
        self.steplen = steplen
        self.n_steps = n_steps + 2
        self.tsp_solver = TSPSolver()
        self.trajectories = np.tile(np.random.rand(self.n_targets, 1, 2), (self.n_steps, 1)) if self.steplen == 0 \
                            else self.run_tsp()  # (n_targets, n_steps, 2)
        self.sigma = np.array([0.1] * self.n_targets)
        self.step = 0
        self.max_value = 1 / (2 * np.pi * self.sigma ** 2)

    def run_tsp(self):
        coord = np.random.rand(self.n_targets, self.n_tsp_nodes, 2)
        for i in range(self.n_targets):
            index = self.tsp_solver.run_solver(coord[i])
            coord[i] = coord[i][index]

        trajectories = []
        for i in range(self.n_targets):
            trajectory = []
            residual_len = 0
            for j in range(self.n_tsp_nodes-1):
                no_step = True
                d = np.linalg.norm(coord[i, j+1, :] - coord[i, j, :])
                remain_len = d
                next_len = self.steplen - residual_len
                while remain_len > next_len:
                    if no_step:
                        pt = (coord[i, j+1, :] - coord[i, j, :]) * next_len / d + coord[i, j, :]
                    else:
                        pt = (coord[i, j+1, :] - coord[i, j, :]) * next_len / d + pt
                    trajectory += [pt]
                    remain_len -= next_len
                    next_len = self.steplen
                    no_step = False
                residual_len = residual_len + remain_len if no_step else remain_len
                if len(trajectory) > self.n_steps:
                    break
            assert len(trajectory) >= self.n_steps
            trajectories += [np.asarray(trajectory)[:self.n_steps]]
        return np.array(trajectories)

    def fn(self, X, update=False):
        if update:
            self.step += 1
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            gaussian_mean = self.trajectories[target_id][self.step]
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

    def plot(self, img0, i=0):
        plt.ion()
        plt.figure()
        n = img0.shape[-1]
        for id, img in enumerate(img0.T):
            plt.subplot(1,n,id+1)
            plt.imshow(img.reshape(40, 40), vmin=0, vmax=1)
            for j in range(self.trajectories.shape[1] - 1):
                plt.plot(self.trajectories[id, j:j+2, 1] * 40, self.trajectories[id, j:j+2, 0] * 40, c='white')
            plt.title(id)
        plt.show()
        plt.pause(0.5)
        plt.close()


class FixedMotion:
    def __init__(self, n_targets=2, n_steps=150):
        self.n_targets = n_targets
        self.n_lines = np.random.randint(0, self.n_targets + 1)
        self.n_circles = self.n_targets - self.n_lines
        self.n_frac = (300, 600)  # decide the movement speed, higher is slower
        self.n_steps = n_steps + 2  # max step
        self.step = 0
        self.sigma = np.array([0.1] * self.n_targets)
        self.trajectories = np.array([self.add_line() for _ in range(self.n_lines)] +
                                     [self.add_circle() for _ in range(self.n_circles)])
        self.max_value = 1 / (2 * np.pi * self.sigma ** 2)

    def add_line(self):
        while True:
            pt1 = np.random.rand(2)
            pt2 = np.random.rand(2)
            if np.linalg.norm(pt1-pt2) > 0.75:  # set min line length
                break
        path = np.linspace(pt1, pt2, num=np.random.randint(*self.n_frac))
        traj = np.vstack((path, path[::-1][1:-1]))
        while True:
            traj = np.vstack((traj, traj))
            if traj.shape[0] >= self.n_steps:
                break
        return traj.tolist()[:self.n_steps]

    def add_circle(self):
        traj = []
        center = np.random.uniform(0.3, 0.7, 2)
        radius = np.random.uniform(0.2, min(*center, *(1-center)))
        theta = np.random.uniform(0, 2*np.pi)
        dtheta = 2*np.pi / np.random.uniform(*self.n_frac)
        is_clockwise = np.random.choice([-1, 1])
        for _ in range(self.n_steps):
            pos = center + radius * np.array([np.sin(theta), np.cos(theta)])
            theta += is_clockwise * dtheta
            traj.append(pos.tolist())
        return traj

    def fn(self, X, update=False):
        if update:
            self.step += 1
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            gaussian_mean = self.trajectories[target_id][self.step]
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

    def plot(self, img0, i=0):
        plt.ion()
        plt.figure()
        n = img0.shape[-1]
        for id, img in enumerate(img0.T):
            plt.subplot(1,n,id+1)
            plt.imshow(img.reshape(30,30), vmin=0, vmax=1)
            plt.scatter(self.trajectories[id, :, 1] * 30, self.trajectories[id, :, 0] * 30, cmap='cyan', marker='.')
            plt.title(i)
        plt.show()
        plt.pause(0.5)
        plt.close()


class FixedPosition:
    def __init__(self, n_targets=2):
        self.n_targets = n_targets
        self.step = 0  # dumb var
        self.sigma = np.array([0.1] * self.n_targets)
        self.mean = np.random.uniform(0, 1, (n_targets, 2))
        self.max_value = 1 / (2 * np.pi * self.sigma**2)

    def fn(self, X, update=False):
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
    # example = DriftGaussian()
    # print(len(example.mean))
    # x1 = np.linspace(0,1)
    # x2 = np.linspace(0,1)
    # x1x2 = np.array(list(product(x1, x2)))
    # for i in range(150):
    #     y = example.fn(x1x2, update=True)

    # example = FixedMotion(2, 100)
    example = VTSPGaussian(2)
    x1 = np.linspace(0,1,40)
    x2 = np.linspace(0,1,40)
    x1x2 = np.array(list(product(x1, x2)))
    for i in range(200):
        y = example.fn(x1x2, steplen=0.01)
