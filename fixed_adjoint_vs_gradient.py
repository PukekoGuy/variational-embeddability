# to generate new data and plot: python adjoint_vs_gradient_time.py
# to run without generating new data: python adjoint_vs_gradient_time.py --plot-only

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from experiment import Observation, trig_generator
from individual_data_adjoint import Embedder as adjoint_Embedder
from individual_data_gradient import Embedder as gradient_Embedder


DEFAULT_RESULTS_FILE = Path(__file__).with_name("adjoint_vs_gradient_results.csv")


def save_results(
    results_file,
    num_individuals,
    adjoint_mise,
    gradient_mise,
    adjoint_execution_time,
    gradient_execution_time,
):
    """Save the MISE and execution-time results as a CSV file."""
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "num_individuals",
                "adjoint_mise",
                "gradient_mise",
                "adjoint_execution_seconds",
                "gradient_execution_seconds",
            ]
        )
        writer.writerows(
            zip(
                num_individuals,
                adjoint_mise,
                gradient_mise,
                adjoint_execution_time,
                gradient_execution_time,
            )
        )


def run_experiments(
    generator,
    rank,
    fixed_times,
    num_intervals,
    num_individuals_range,
    results_file=DEFAULT_RESULTS_FILE,
    mu=1e-2,
    kappa=1.0,
    max_iter=500,
):
    """Simulate and fit both models for several panel sizes.

    Each call to an embedder's ``optimise_A`` method is timed separately using
    a monotonic high-resolution clock. The MISE and timing results are written
    to ``results_file`` after all experiments finish.

    Returns
    -------
    tuple
        ``(true_A, adjoint_estimated_paths, gradient_estimated_paths, adjoint_mise,
        gradient_mise, adjoint_execution_time, gradient_execution_time)``.
    """
    num_individuals_values = np.asarray(list(num_individuals_range), dtype=int)
    if num_individuals_values.size == 0:
        raise ValueError("num_individuals_range must contain at least one value")

    true_A = np.stack(
        [generator(k / num_intervals) for k in range(num_intervals + 1)]
    )
    adjoint_estimated_paths = []
    gradient_estimated_paths = []
    adjoint_mise_values = []
    gradient_mise_values = []
    adjoint_execution_times = []
    gradient_execution_times = []

    adjoint_emb = adjoint_Embedder(
        rank,
        num_intervals=num_intervals,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )
    gradient_emb = gradient_Embedder(
        rank,
        num_intervals=num_intervals,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )

    for num_individuals in num_individuals_values:
        observations = Observation(generator, rank, fixed_times, num_intervals)
        observed_paths = observations.simulate_fixed_observations(num_individuals)

        start = perf_counter()
        adjoint_estimated_A = adjoint_emb.optimise_A(fixed_times, observed_paths)
        adjoint_execution_times.append(perf_counter() - start)

        start = perf_counter()
        gradient_estimated_A = gradient_emb.optimise_A(fixed_times, observed_paths)
        gradient_execution_times.append(perf_counter() - start)

        # Store the actual estimates (the previous version accidentally stored
        # the Embedder classes here).
        adjoint_estimated_paths.append(adjoint_estimated_A)
        gradient_estimated_paths.append(gradient_estimated_A)
        adjoint_mise_values.append(np.mean((true_A - adjoint_estimated_A) ** 2))
        gradient_mise_values.append(np.mean((true_A - gradient_estimated_A) ** 2))

        print(
            f"n={num_individuals}: "
            f"adjoint={adjoint_execution_times[-1]:.3f}s, "
            f"gradient={gradient_execution_times[-1]:.3f}s"
        )

    adjoint_mise_values = np.asarray(adjoint_mise_values)
    gradient_mise_values = np.asarray(gradient_mise_values)
    adjoint_execution_times = np.asarray(adjoint_execution_times)
    gradient_execution_times = np.asarray(gradient_execution_times)

    save_results(
        results_file,
        num_individuals_values,
        adjoint_mise_values,
        gradient_mise_values,
        adjoint_execution_times,
        gradient_execution_times,
    )

    return (
        true_A,
        adjoint_estimated_paths,
        gradient_estimated_paths,
        adjoint_mise_values,
        gradient_mise_values,
        adjoint_execution_times,
        gradient_execution_times,
    )


def generate_data(
    results_file=DEFAULT_RESULTS_FILE,
    target_jumps=1,
    num_observations=20,
    rank=2,
    num_intervals=100,
    num_individuals_range=range(500, 10001, 500),
    max_iter=500,
):
    """Run the original adjoint versus gradient experiment and save it."""
    observations = Observation(trig_generator(), rank)
    observations.tune_generator_for_target_jumps(
        lambda A: trig_generator(A=A), target_jumps
    )

    # Retain this calculation from the original experiment so the tuned
    # generator's expected number of jumps can be inspected while debugging.
    observations.expected_jumps(np.full(rank, 1.0 / rank))

    times = np.linspace(0.0, 1.0, num_observations)
    return run_experiments(
        observations.generator,
        rank,
        times,
        num_intervals,
        num_individuals_range,
        results_file=results_file,
        max_iter=max_iter,
    )


def plot_results_from_file(results_file=DEFAULT_RESULTS_FILE, figure_file=None, show=True):
    """Read a saved results CSV and plot MISE and execution time.

    This function performs no simulation or optimisation, so it can be called
    repeatedly to adjust or reproduce the plots without regenerating the data.
    """
    results_file = Path(results_file)
    data = np.genfromtxt(results_file, delimiter=",", names=True, encoding="utf-8")
    data = np.atleast_1d(data)

    num_individuals = data["num_individuals"]
    adjoint_mise = data["adjoint_mise"]
    gradient_mise = data["gradient_mise"]
    adjoint_execution_time = data["adjoint_execution_seconds"]
    gradient_execution_time = data["gradient_execution_seconds"]

    log_x = np.log(num_individuals)
    adjoint_slope = linregress(log_x, np.log(adjoint_mise)).slope
    gradient_slope = linregress(log_x, np.log(gradient_mise)).slope
    mise_summary = (
        f"Adjoint slope = {adjoint_slope:.3f}, "
        f"Gradient slope = {gradient_slope:.3f}"
    )
    print("mise summary:", mise_summary)

    mise_title = "log(MISE) vs. log(number of individuals)"

    fig, (mise_ax, time_ax) = plt.subplots(1, 2, figsize=(12, 5))

    mise_ax.loglog(
        num_individuals,
        adjoint_mise,
        marker="o",
        color="tab:blue",
        label="adjoint",
    )
    mise_ax.loglog(
        num_individuals,
        gradient_mise,
        marker="o",
        color="tab:orange",
        label="gradient",
    )
    mise_ax.set_xlabel("Number of individuals sampled")
    mise_ax.set_ylabel("Mean integrated square error")
    mise_ax.set_title(mise_title)
    mise_ax.grid(True, which="both", alpha=0.3)
    mise_ax.legend()

    time_ax.plot(
        num_individuals,
        adjoint_execution_time,
        marker="o",
        color="tab:blue",
        label="adjoint",
    )
    time_ax.plot(
        num_individuals,
        gradient_execution_time,
        marker="o",
        color="tab:orange",
        label="gradient",
    )
    time_ax.set_xlabel("Number of individuals sampled")
    time_ax.set_ylabel("Execution time (seconds)")
    time_ax.set_title("Execution Time vs. Number of Individuals")
    time_ax.grid(True, alpha=0.3)
    time_ax.legend()

    fig.tight_layout()

    if figure_file is not None:
        figure_file = Path(figure_file)
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, (mise_ax, time_ax)


def main():
    parser = argparse.ArgumentParser(
        description="Compare adjoint and gradient embedding experiments."
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="CSV file used to save or load the experimental results.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read and plot existing data without rerunning the optimisations.",
    )
    parser.add_argument(
        "--figure-file",
        type=Path,
        help="Optional path at which to save the figure.",
    )
    args = parser.parse_args()

    if not args.plot_only:
        generate_data(args.results_file)
    plot_results_from_file(args.results_file, figure_file=args.figure_file)


if __name__ == "__main__":
    main()
