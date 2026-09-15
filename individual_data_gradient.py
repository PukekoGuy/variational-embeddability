import numpy as np
from scipy.optimize import minimize
from experiment import *
from matrix_helpers import *

class Embedder:
    def __init__(self, rank, num_intervals=100, mu=1e-2, kappa=1e2, max_iter=500):
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals
        self.mu = mu
        self.kappa = kappa
        self.rank = rank
        self.num_rates = rank * (rank - 1)
        self.epsilon = 1e-12
        self.maxiter = max_iter

    def optimise_A(self, obs_times, obs_data):
        N = self.num_intervals + 1
        x0 = np.full((N, self.num_rates), 1.0).ravel()
        obs_times = self.times_to_intervals(obs_times)

        max_rate = 0.99 / (self.dt * (self.rank - 1))
        res = minimize(
            lambda z: self.objective_grad(z, obs_times, obs_data),
            x0,
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, disp=False),
            bounds=[(0.0, max_rate)] * (N * self.num_rates),
        )

        theta = res.x.reshape(N, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)

        return A 

    def objective_grad(self, x, obs_times, obs_data):
        theta = x.reshape(self.num_intervals + 1, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)

        U = 0.0
        matrix_gradient = np.zeros_like(A)

        for k in range(1, len(obs_times)):
            U += self.objective_grad_on_interval(
                A, obs_times, obs_data, k, matrix_gradient
            )

        differences = A[1:] - A[:-1]
        U += 0.5 * self.mu / self.dt * np.sum(differences**2)

        matrix_gradient[1:] += self.mu / self.dt * differences
        matrix_gradient[:-1] -= self.mu / self.dt * differences

        G = np.stack([
            matrix_to_rates(grad, self.rank, self.num_rates)
            for grad in matrix_gradient
        ])

        return U, G.ravel()

    def objective_grad_on_interval(self, A, obs_times, obs_data, k, G):
        t0 = obs_times[k - 1]
        t1 = obs_times[k]
        length = t1 - t0
        I = np.eye(self.rank)

        factors = [I + self.dt * A[tau] for tau in range(t0, t1)]

        forward = [I]
        for factor in factors:
            forward.append(forward[-1] @ factor)

        backward = [I for _ in range(length + 1)]
        for i in range(length - 1, -1, -1):
            backward[i] = factors[i] @ backward[i + 1]

        transition = forward[-1]
        starts = obs_data[k - 1]
        ends = obs_data[k]

        counts = np.zeros_like(transition)
        np.add.at(counts, (starts, ends), 1.0)

        U = -self.kappa * np.sum(counts * np.log(transition + self.epsilon))
        M = counts / (transition + self.epsilon)

        for i, tau in enumerate(range(t0, t1)):
            G[tau] -= (
                self.kappa
                * self.dt
                * forward[i].T
                @ M
                @ backward[i + 1].T
            )

        return U

    def times_to_intervals(self, times):
        return [int(round(t * self.num_intervals)) for t in times]


def run_experiments(gen, times, num_intervals, num_individuals_range, max_iter=500, show_plot = False):
    gen_paths = []
    embedder = Embedder(2, num_intervals, mu=1e-2, kappa=1.0, max_iter=max_iter)
    MISE = []
    generator = trig_generator(1.0)
    true_A = np.stack([generator(t/num_intervals) for t in range(num_intervals+1)])
    for num_individuals in num_individuals_range:
        obs = Observation(gen, 2, times=times, num_intervals=num_intervals)
        paths = obs.simulate_fixed_observations(num_individuals)
        A = embedder.optimise_A(times, paths)
        gen_paths.append(A)

        # MISE between true and estimated A matrices
        MISE.append(np.sum((true_A - A) ** 2) / len(A))
        print(MISE[-1])

    if show_plot:
        plot_MISE(num_individuals_range, MISE)

    return gen_paths, true_A


# times = np.linspace(0.0, 1.0, 21)
# gen = trig_generator(1.0)
# np.random.seed(1008)
# long_range = range(100, 10001, 100)
# medium_range = range(500,5001,500)
# test_range = range(500, 1501, 500)
# gen_paths, true_A = run_experiments(gen, times, 100, test_range, max_iter=500)
# plot_2D_generator_path(gen_paths, true_A, test_range, times, plot_frequency=1)

