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

    def exponentials(self, A):
        return [
            np.eye(self.rank) + self.dt*A[k] #expm(self.dt*A)
            for k in range(self.num_intervals)
        ]

    def objective_grad(self, x, obs_times, obs_data):

        theta = x.reshape(self.num_intervals + 1, self.num_rates)

        A = A_mat_from_rates(theta, self.rank)
        E = self.exponentials(A)

        U = 0.0
        rate_gradient = np.zeros_like(theta)

        for r in range(len(obs_times) - 1):

            a = obs_times[r]
            b = obs_times[r + 1]

            # Transition counts from observation r -> r+1
            starts = obs_data[r]
            ends = obs_data[r + 1]

            counts = np.zeros((self.rank, self.rank))
            np.add.at(counts, (starts, ends), 1.0)

            # Local forward states:
            #
            # prefix[q] = E_a ... E_{a+q-1}
            #
            # so prefix[0] = I
            prefix = [np.eye(self.rank)]

            for k in range(a, b):
                prefix.append(prefix[-1] @ E[k])

            transition = prefix[-1]

            # Negative log likelihood
            U -= self.kappa * np.sum(
                counts * np.log(transition + self.epsilon)
            )

            # Terminal adjoint:
            #
            # dJ/dT = -kappa * counts / T
            Lam = -self.kappa * counts / (
                transition + self.epsilon
            )

            # Backward adjoint
            for k in range(b - 1, a - 1, -1):

                Fk = prefix[k - a]

                # dJ/dA_k = dt F_k^T Lambda_{k+1}
                M = self.dt * Fk.T @ Lam

                rate_gradient[k] += adjoint_to_rates(M, self.rank, self.num_rates)

                # Lambda_k = Lambda_{k+1} E_k^T
                Lam = Lam @ E[k].T

        differences = theta[1:] - theta[:-1]

        U += (
            0.5 * self.mu / self.dt
            * np.sum(differences ** 2)
        )

        rate_gradient[1:] += (
            self.mu / self.dt * differences
        )

        rate_gradient[:-1] -= (
            self.mu / self.dt * differences
        )

        return U, rate_gradient.ravel()
    
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
        #P = self.forward(A)

        return A #, P, res.fun

    def times_to_intervals(self, times):
        return [int(round(t * self.num_intervals)) for t in times]

# def run_experiments(gen, times, num_intervals, num_individuals_range, max_iter=500, show_plot = False):
#     from experiment import Observation

#     true_A = np.stack(
#         [generator(t/num_intervals) for t in range(num_intervals+1)]
#     )

#     estimated = []
#     embedder = Embedder(2, num_intervals, mu=1e-2, kappa=1.0, max_iter=max_iter)
#     MISE = []
#     generator = trig_generator(1.0)
    
#     for num_individuals in num_individuals_range:
#         obs = Observation(gen, 2, times=times, num_intervals=num_intervals)
#         paths = obs.simulate_fixed_observations(num_individuals)
#         A = embedder.optimise_A(times, paths)
#         estimated.append(A)

#         # MISE between true and estimated A matrices
#         MISE.append(np.sum((true_A - A) ** 2) / len(A))

#     if show_plot:
#         plot_MISE(num_individuals_range, MISE)

#     return estimated, true_A

def run_experiments(
    generator,
    rank,
    target_num_jumps,
    num_observations,
    num_intervals,
    num_individuals_range,
    mu=1e-2,
    kappa=1.0,
    max_iter=500,
):
    """Simulate and fit the model for several panel sizes.

    Returns ``(estimated_paths, true_path, mise_values)``. Unlike the old
    two-state helper, this works for every rank and includes every level's
    distinct birth and death rates in the MISE.
    """
    print(num_individuals_range)

    from experiment import Observation

    estimated_paths = []
    mise_values = []
    embedder = Embedder(
        rank,
        num_intervals=num_intervals,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )

    times = np.linspace(0.0, 1.0, num_observations)
    observations = Observation(generator, rank, times, num_intervals)
    observations.tune_generator_for_target_jumps(
        lambda A: trig_generator(A=A), target_num_jumps
    )
    generator = observations.generator

    true_A = np.stack(
        [generator(k / num_intervals) for k in range(num_intervals + 1)]
    )

    for num_individuals in num_individuals_range:
        print(int(num_individuals))
        paths = observations.simulate_fixed_observations(int(num_individuals))
        estimated_A = embedder.optimise_A(times, paths)
        estimated_paths.append(estimated_A)
        mise_values.append(np.mean((true_A - estimated_A) ** 2))

    return estimated_paths, true_A, np.asarray(mise_values)

# gen = trig_generator()
# np.random.seed(1008)
# long_range = range(100, 10001, 100)
# medium_range = np.linspace(500,5000,10)
# test_range = range(500, 1501, 500)
# gen_paths, true_A, mise = run_experiments(gen, 2, num_jumps=1, num_observations=20, num_intervals=100, num_individuals_range=medium_range, max_iter=500)
# plot_2D_generator_path(gen_paths, true_A, medium_range, plot_frequency=1)

# times = [0.,1.]
# paths = np.array([
#     [0,0,1,1],
#     [1,0,1,0]
# ])
# embedder = Embedder(2, 100, mu=1e-2, kappa=1.0, max_iter=500)
# A = embedder.optimise_A(times, paths)
# plot_2D_generator_path([A], A, [0], times, plot_frequency=1)
