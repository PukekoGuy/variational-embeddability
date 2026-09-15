import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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

    def expected_jumps(self, initial_distribution, num_increments=100):
        integrand = lambda t: initial_distribution[None, :].T @ np.diagonal(self.generator(t))[None, :] * self.P(0,t) 
        integral = -sum(
            1/num_increments * integrand(s/num_increments) for s in range(num_increments)
        )
        return integral.sum()

    def tune_generator_for_target_jumps(self, generator_with_parameter, target_num_jumps, initial_guess = 1, initial_distribution=None):
        if initial_distribution is None:
            initial_distribution = np.array([1/self.rank] * self.rank)

        def expected_jumps_from_parameter(parameter):
            parameter = float(parameter)
            self.generator = generator_with_parameter(parameter)
            return self.expected_jumps(initial_distribution) - target_num_jumps
        
        solution = fsolve(expected_jumps_from_parameter, initial_guess)
        return solution

    def simulate_fixed_observations(self, num_individuals, initial_states=None):
        paths = np.zeros((len(self.times), num_individuals), dtype=int)

        if initial_states is None:
            for j in range(num_individuals):
                paths[0, j] = np.random.choice(self.rank)

        for j in range(num_individuals):
            state = paths[0, j]

            for k in range(1, len(self.times)):
                P = self.P(self.times[k - 1], self.times[k])
                state = np.random.choice(self.rank, p=P[state])
                paths[k, j] = state

        return paths

    def simulate_individual_jump_times_observations(self, num_individuals, initial_states=None):
        """Simulate the Euler-discretised chain and store only actual changes.

        jump_times are interval indices t in {0,...,num_intervals-1}.
        states[m] has one more element than jump_times[m].
        """
        jump_times = []
        states = []
        num_jumps = np.zeros(num_individuals)

        if initial_states is None:
            for _ in range(num_individuals):
                states.append([np.random.choice(self.rank)])

        for m in range(num_individuals):
            current_state = states[m][0]
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

                    num_jumps[m] += 1

        return jump_times, states


def birth_death_generator(rank, p, q, t):
    G = np.zeros((rank, rank))
    for i in range(rank-1):
        G[i,i+1] = p(t)
        G[i,i] -= p(t)
        G[i+1,i] = q(t)
        G[i+1,i+1] -= p(t)


def trig_generator(f=1, A=1):
    return lambda t: np.array([
        [A*(-1 - np.cos(2 * np.pi * f * t)), A*(1 + np.cos(2 * np.pi * f * t))],
        [A*(1 + np.sin(2 * np.pi * f * t)), A*(-1 - np.sin(2 * np.pi * f * t))],
    ])

def plot_MISE(x_range, MISE):
    from scipy.stats import linregress

    log_x = np.log(x_range)
    log_y = np.log(MISE)
    line = linregress(log_x,log_y)
    slope = line.slope
    r = np.corrcoef(log_x,log_y)[0,1]

    fig, ax = plt.subplots()
    ax.loglog(x_range, MISE)
    ax.set_ylabel("Mean integrated square error")
    ax.set_xlabel("Number of individuals sampled")
    ax.set_title(f'Slope = {slope}')
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def plot_2D_generator_path(
    gen_paths, true_A, num_individuals_range, times=None, plot_frequency=1,
    *, ax=None, return_plot=False, title=None, fontsize=14, observation_markers=False
):
    """Plot generator paths, optionally into an existing subplot.

    Pass ``ax`` to draw into a subplot without showing the figure, or set
    ``return_plot=True`` to return a new Axes without showing it. Both options
    return the Axes for further customization. ``title=None`` uses the default
    title; pass an empty string to omit it. ``fontsize`` controls labels,
    ticks, and the legend, with the title two points larger.
    """
    assert gen_paths[0][0].shape == (2, 2)

    supplied_ax = ax is not None
    if ax is None:
        _, ax = plt.subplots()

    num_individuals_to_plot = list(num_individuals_range)[::plot_frequency]
    gen_paths_to_plot = gen_paths[::plot_frequency]
    print(len(num_individuals_to_plot))

    true_x = [true_A[t][0, 1] for t in range(len(true_A))]
    true_y = [true_A[t][1, 0] for t in range(len(true_A))]

    ax.plot(true_x, true_y, label="true", color="C1", linestyle="--")

    cmap = plt.colormaps["viridis"]
    denom = max(1, len(num_individuals_to_plot) - 1)
    for k, (num_individuals, gen_path) in enumerate(zip(num_individuals_to_plot, gen_paths_to_plot)):
        x = [gen_path[t][0, 1] for t in range(len(gen_path))]
        y = [gen_path[t][1, 0] for t in range(len(gen_path))]
        color = cmap(k / denom)
        ax.plot(x, y, label=f"{num_individuals}", color=color, alpha=0.5)

    if (times is not None) and observation_markers:
        for t in times:
            T = int(round(t * (len(gen_paths[0]) - 1)))
            ax.plot(true_x[T], true_y[T], marker="x", linestyle="None", color="black", markersize=8, mew=2)

    ax.set_xlabel(r"$A(t)_{12}$", fontsize=fontsize)
    ax.set_ylabel(r"$A(t)_{21}$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.axis("scaled")
    ax.set_title(
        "Estimated vs true curve from matrix entries" if title is None else title,
        fontsize=fontsize + 2,
    )
    ax.grid(True)
    if len(num_individuals_to_plot) < 5:
        ax.legend(fontsize=fontsize)
    if return_plot or supplied_ax:
        return ax
    plt.show()
