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

        # --------------------------------------------------
        # DATA LIKELIHOOD + ADJOINT, ONE OBSERVATION
        # INTERVAL AT A TIME
        # --------------------------------------------------

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

        # --------------------------------------------------
        # SMOOTHNESS PENALTY
        # --------------------------------------------------

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

# def run_experiments(gen, times, num_intervals, num_individuals_range, max_iter=500, show_plot=False):
#     gen_paths = []
#     embedder = Embedder(2, num_intervals, mu=1e-2, kappa=1.0, max_iter=max_iter)
#     MISE = []
#     generator = trig_generator(1.0)
#     true_A = np.stack([generator(t/num_intervals) for t in range(num_intervals+1)])
#     for num_individuals in num_individuals_range:
#         obs = Observation(gen, 2, times, num_intervals)
#         paths = obs.simulate(num_individuals)
#         A = embedder.optimise_A(times, paths)
#         gen_paths.append(A)
#         # print(A)

#         # MISE between true and estimated A matrices
#         MISE.append(np.sum((true_A - A) ** 2) / len(A))
#         print(MISE[-1])

#     if show_plot:
#         plt.plot(num_individuals_range, MISE)
#         plt.ylabel("Mean integrated square error")
#         plt.xlabel("Number of individuals sampled")
#         plt.title("MISE of the estimated generator path vs number of individuals")
#         plt.grid(True)
#         plt.show()

#     return gen_paths, true_A

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

times = np.linspace(0.0, 1.0, 21)
gen = trig_generator(1.0)
np.random.seed(1008)
long_range = range(100, 10001, 100)
medium_range = range(500,5001,500)
test_range = range(500, 1501, 500)
gen_paths, true_A = run_experiments(gen, times, 100, test_range, max_iter=500)
plot_2D_generator_path(gen_paths, true_A, test_range, times, plot_frequency=1)




