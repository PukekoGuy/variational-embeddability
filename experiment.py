import numpy as np
import matplotlib.pyplot as plt


class Observation:
    def __init__(self, generator, rank, times=None, num_intervals=100):
        self.generator = generator
        self.rank = rank
        self.times = times
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals

    def P(self, s, t):
        S = int(round(s * self.num_intervals))
        T = int(round(t * self.num_intervals))
        P = np.eye(self.rank)

        for tau in range(S, T):
            r = tau / self.num_intervals
            P = P @ (np.eye(self.rank) + self.dt * self.generator(r))

        return P

    def simulate_fixed_observations(self, num_individuals):
        paths = np.zeros((len(self.times), num_individuals), dtype=int)

        for j in range(num_individuals):
            state = np.random.choice(self.rank)
            paths[0, j] = state

            for k in range(1, len(self.times)):
                P = self.P(self.times[k - 1], self.times[k])
                state = np.random.choice(self.rank, p=P[state])
                paths[k, j] = state

        return paths

    def simulate_individual_jump_times_observations(self, num_individuals):
        """Simulate the Euler-discretised chain and store only actual changes.

        jump_times are interval indices t in {0,...,num_intervals-1}.
        states[m] has one more element than jump_times[m].
        """
        jump_times = []
        states = []

        for m in range(num_individuals):
            current_state = np.random.choice(self.rank)
            states.append([current_state])
            jump_times.append([])

            for t in range(self.num_intervals):
                # FIX: generator expects physical time in [0,1], not the interval index.
                physical_time = t / self.num_intervals
                P = np.eye(self.rank) + self.dt * self.generator(physical_time)
                new_state = np.random.choice(self.rank, p=P[current_state])

                if new_state != current_state:
                    states[m].append(new_state)
                    current_state = new_state
                    jump_times[m].append(t)

        return jump_times, states


def trig_generator(f):
    return lambda t: np.array([
        [-1 - np.cos(2 * np.pi * f * t), 1 + np.cos(2 * np.pi * f * t)],
        [1 + np.sin(2 * np.pi * f * t), -1 - np.sin(2 * np.pi * f * t)],
    ])


def plot_MISE(x_range, MISE):
    plt.plot(x_range, MISE)
    plt.ylabel("Mean integrated square error")
    plt.xlabel("Number of individuals sampled")
    plt.title("MISE of the estimated generator path vs number of individuals")
    plt.grid(True)
    plt.show()


def plot_2D_generator_path(gen_paths, true_A, num_individuals_range, times=None, plot_frequency=1):
    assert gen_paths[0][0].shape == (2, 2)

    num_individuals_to_plot = list(num_individuals_range)[::plot_frequency]
    gen_paths_to_plot = gen_paths[::plot_frequency]

    true_x = [true_A[t][0, 1] for t in range(len(true_A))]
    true_y = [true_A[t][1, 0] for t in range(len(true_A))]

    plt.plot(true_x, true_y, label="true", color="C1", linestyle="--")

    cmap = plt.colormaps["viridis"]
    denom = max(1, len(num_individuals_to_plot) - 1)
    for k, (num_individuals, gen_path) in enumerate(zip(num_individuals_to_plot, gen_paths_to_plot)):
        x = [gen_path[t][0, 1] for t in range(len(gen_path))]
        y = [gen_path[t][1, 0] for t in range(len(gen_path))]
        color = cmap(k / denom)
        plt.plot(x, y, label=f"{num_individuals}", color=color, alpha=0.5)

    if times is not None:
        for t in times:
            T = int(round(t * (len(gen_paths[0]) - 1)))
            plt.plot(true_x[T], true_y[T], marker="x", linestyle="None", color="black", markersize=8, mew=2)

    plt.xlabel(r"$A(t)_{12}$")
    plt.ylabel(r"$A(t)_{21}$")
    plt.axis("scaled")
    plt.title("Estimated vs true curve from matrix entries")
    plt.grid(True)
    if len(num_individuals_to_plot) < 5:
        plt.legend()
    plt.show()