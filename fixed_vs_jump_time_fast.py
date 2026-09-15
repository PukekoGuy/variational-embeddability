"""Fixed versus jump-time experiments with batched simulation.

Run: python fixed_vs_jump_time_fast.py
Quick run: python fixed_vs_jump_time_fast.py --individuals 100 1000 --max-iter 20
Plot saved data: python fixed_vs_jump_time_fast.py --plot-only

The Euler transition model and fitting implementations are unchanged. Random
draws are batched, so a seed does not reproduce the original script's samples.
Only compressed jump paths are stored, not an individuals-by-time array.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from experiment import Observation, trig_generator
from individual_data_adjoint import Embedder as Fixed_Embedder
from individual_jumps_gradient import Embedder as Jump_Embedder

DEFAULT_RESULTS_FILE = Path(__file__).with_name("fixed_vs_jump_time_fast_results.csv")
FIELDS = (
    "num_individuals", "fixed_mise", "jump_mise",
    "fixed_execution_seconds", "jump_execution_seconds",
    "fixed_simulation_seconds", "jump_simulation_seconds", "iteration_seconds",
)


class BatchedObservation:
    """Cache transitions once, then sample independent individuals in batches.

    As in Observation, jump times are interval indices, and fixed observation
    times are rounded to the nearest grid point. The generator must be a
    deterministic function of time. Initial states default to uniform draws.
    """

    def __init__(self, generator, rank, times, num_intervals=1000, seed=None):
        if rank < 1 or num_intervals < 1:
            raise ValueError("rank and num_intervals must be positive")
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) < 0) or times[0] < 0 or times[-1] > 1):
            raise ValueError("times must be sorted within [0, 1], with at least two entries")
        self.rank = rank
        self.num_intervals = num_intervals
        self.rng = np.random.default_rng(seed)
        self.true_A = np.stack([generator(k / num_intervals)
                                for k in range(num_intervals + 1)])
        if self.true_A.shape != (num_intervals + 1, rank, rank):
            raise ValueError("generator matrix shape must match rank")
        steps = np.eye(rank)[None, :, :] + self.true_A[:-1] / num_intervals
        indices = np.rint(times * num_intervals).astype(int)
        self.fixed_transitions = []
        for a, b in zip(indices[:-1], indices[1:]):
            product = np.eye(rank)
            for step in steps[a:b]:
                product = product @ step
            self.fixed_transitions.append(product)
        self.jump_cdf = self._cdf(steps)
        self.fixed_cdf = self._cdf(np.asarray(self.fixed_transitions))

    @staticmethod
    def _cdf(transitions):
        if (not np.all(np.isfinite(transitions)) or np.any(transitions < 0)
                or not np.allclose(transitions.sum(axis=-1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Invalid Euler probabilities; increase num_intervals or check generator")
        # Normalisation removes accumulated roundoff in matrix products.
        probabilities = transitions / transitions.sum(axis=-1, keepdims=True)
        return np.cumsum(probabilities, axis=-1)[..., :-1]

    def _initial_states(self, n, initial_states):
        if n < 1:
            raise ValueError("num_individuals must be positive")
        if initial_states is None:
            return self.rng.integers(self.rank, size=n)
        states = np.asarray(initial_states)
        if (states.shape != (n,) or not np.issubdtype(states.dtype, np.integer)
                or np.any(states < 0) or np.any(states >= self.rank)):
            raise ValueError("initial_states must contain one valid integer state per individual")
        return states.copy()

    def _draw(self, cdf, states):
        # One uniform per individual; inverse categorical CDF, for any rank.
        return np.sum(self.rng.random(states.size)[:, None] >= cdf[states], axis=1)

    def simulate_fixed_observations(self, num_individuals, initial_states=None):
        states = self._initial_states(num_individuals, initial_states)
        paths = np.empty((len(self.fixed_cdf) + 1, num_individuals), dtype=int)
        paths[0] = states
        for k, cdf in enumerate(self.fixed_cdf, start=1):
            states = self._draw(cdf, states)
            paths[k] = states
        return paths

    def simulate_individual_jump_times_observations(
        self, num_individuals, initial_states=None, batch_size=10000,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        initial = self._initial_states(num_individuals, initial_states)
        jump_times = [[] for _ in range(num_individuals)]
        states = [[int(state)] for state in initial]
        for start in range(0, num_individuals, batch_size):
            current = initial[start:start + batch_size].copy()
            for t, cdf in enumerate(self.jump_cdf):
                new = self._draw(cdf, current)
                # Python work is needed only for actual changes, not every step.
                for local in np.flatnonzero(new != current):
                    individual = start + int(local)
                    jump_times[individual].append(t)
                    states[individual].append(int(new[local]))
                current = new
        return jump_times, states


def run_experiments(
    generator, rank, fixed_times, num_intervals, num_individuals_list,
    results_file=DEFAULT_RESULTS_FILE, mu=1e-2, kappa=1.0, max_iter=500,
    seed=None, batch_size=10000,
):
    """Fit the existing embedders; save each completed iteration's timings.

    Execution columns retain their original meaning: optimise_A only (including
    its data preparation). Iteration time includes simulation, fitting and MISE,
    but excludes CSV writing and printing. Shared setup is reported separately.
    The returned seven-tuple matches the original run_experiments function.
    """
    sizes = np.asarray(list(num_individuals_list), dtype=int)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 1):
        raise ValueError("num_individuals_list must contain positive sample sizes")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    start = perf_counter()
    observations = BatchedObservation(generator, rank, fixed_times, num_intervals, seed)
    fixed_emb = Fixed_Embedder(rank, num_intervals, mu, kappa, max_iter)
    jump_emb = Jump_Embedder(rank, num_intervals, mu, kappa, max_iter)
    print(f"Shared transition setup: {perf_counter() - start:.3f}s", flush=True)
    true_A = observations.true_A
    fixed_paths, jump_paths, rows = [], [], []
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for n in sizes:
            print(f"n={n}: simulating and fitting...", flush=True)
            iteration_start = perf_counter()
            start = perf_counter()
            fixed_data = observations.simulate_fixed_observations(n)
            fixed_simulation = perf_counter() - start
            start = perf_counter()
            jump_times, jump_states = observations.simulate_individual_jump_times_observations(
                n, batch_size=batch_size,
            )
            jump_simulation = perf_counter() - start
            start = perf_counter()
            fixed_A = fixed_emb.optimise_A(fixed_times, fixed_data)
            fixed_fit = perf_counter() - start
            start = perf_counter()
            jump_A = jump_emb.optimise_A(jump_times, jump_states)
            jump_fit = perf_counter() - start
            fixed_paths.append(fixed_A)
            jump_paths.append(jump_A)
            row = dict(zip(FIELDS[:-1], (
                int(n), np.mean((true_A - fixed_A)**2), np.mean((true_A - jump_A)**2),
                fixed_fit, jump_fit, fixed_simulation, jump_simulation,
            )))
            # Free the old panel before allocating the next, larger one.
            del fixed_data, jump_times, jump_states
            row["iteration_seconds"] = perf_counter() - iteration_start
            rows.append(row)
            writer.writerow(row)
            output.flush()
            print(
                f"n={n}: fixed-simulation={fixed_simulation:.3f}s, "
                f"jump-simulation={jump_simulation:.3f}s, "
                f"fixed-fit={fixed_fit:.3f}s, jump-fit={jump_fit:.3f}s, "
                f"total={row['iteration_seconds']:.3f}s", flush=True,
            )
    return (true_A, fixed_paths, jump_paths,
            *(np.asarray([row[key] for row in rows]) for key in FIELDS[1:5]))


def generate_data(
    results_file=DEFAULT_RESULTS_FILE, target_jumps=8, rank=2,
    num_intervals=1000, num_individuals_list=None, max_iter=1000,
    seed=None, batch_size=10000,
):
    if rank != 2:
        raise ValueError("trig_generator is rank 2; use run_experiments for other generators")
    if num_individuals_list is None:
        num_individuals_list = np.rint(np.geomspace(1, 1e6, 6)).astype(int)
    start = perf_counter()
    observations = Observation(trig_generator(), rank)
    observations.tune_generator_for_target_jumps(lambda A: trig_generator(A=A), target_jumps)
    print(f"Generator tuning: {perf_counter() - start:.3f}s", flush=True)
    return run_experiments(
        observations.generator, rank, np.linspace(0, 1, target_jumps),
        num_intervals, num_individuals_list, results_file=results_file,
        max_iter=max_iter, seed=seed, batch_size=batch_size,
    )


def plot_results_from_file(results_file=DEFAULT_RESULTS_FILE, figure_file=None, show=True):
    data = np.atleast_1d(np.genfromtxt(results_file, delimiter=",", names=True))
    fig, (mise_ax, time_ax) = plt.subplots(1, 2, figsize=(13, 5))
    n = data["num_individuals"]
    for name, color in (("fixed", "tab:blue"), ("jump", "tab:orange")):
        mise_ax.loglog(n, data[f"{name}_mise"], "o-", color=color, label=name)
        time_ax.plot(n, data[f"{name}_execution_seconds"], "o-", color=color,
                     label=f"{name} fit")
        # time_ax.plot(n, data[f"{name}_simulation_seconds"], "o--", color=color,
        #              label=f"{name} simulation")
    # time_ax.plot(n, data["iteration_seconds"], "o-", color="black", label="iteration total")
    mise_ax.set(ylabel="Mean integrated square error", title="MISE vs. Number of Individuals")
    time_ax.set_xscale('log')
    time_ax.set(ylabel="Elapsed time (seconds)", title="Simulation, Fitting and Total Time")
    for ax in (mise_ax, time_ax):
        ax.set_xlabel("Number of individuals sampled")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
    fig.tight_layout()
    if figure_file is not None:
        figure_file = Path(figure_file)
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (mise_ax, time_ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--figure-file", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--individuals", type=int, nargs="+",
                        help="Sample sizes; default: six log-spaced sizes from 1 to 1,000,000")
    parser.add_argument("--num-intervals", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not args.plot_only:
        generate_data(args.results_file, num_individuals_list=args.individuals,
                      num_intervals=args.num_intervals, max_iter=args.max_iter,
                      seed=args.seed, batch_size=args.batch_size)
    plot_results_from_file(args.results_file, args.figure_file, show=not args.no_show)


if __name__ == "__main__":
    main()
