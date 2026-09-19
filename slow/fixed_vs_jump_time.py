# to generate new data and plot: python fixed_vs_jump_time.py
# to run without generating new data: python fixed_vs_jump_time.py --plot-only

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from experiment import Observation, trig_generator, plot_2D_generator_path
from cosmetics import LABEL_FONT_SIZE, LEGEND_FONT_SIZE, TICK_LABEL_FONT_SIZE, TITLE_FONT_SIZE
from individual_data_adjoint import Embedder as Fixed_Embedder
from individual_jumps_gradient import Embedder as Jump_Embedder

DEFAULT_RESULTS_FILE = Path(__file__).with_name("fixed_vs_jump_time_results.csv")


def save_results(
    results_file,
    num_individuals,
    fixed_mise,
    jump_mise,
    fixed_execution_time,
    jump_execution_time,
):
    """Save the MISE and execution-time results as a CSV file."""
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "num_individuals",
                "fixed_mise",
                "jump_mise",
                "fixed_execution_seconds",
                "jump_execution_seconds",
            ]
        )
        writer.writerows(
            zip(
                num_individuals,
                fixed_mise,
                jump_mise,
                fixed_execution_time,
                jump_execution_time,
            )
        )


def run_experiments(
    generator,
    rank,
    fixed_times,
    num_intervals,
    num_individuals_list,
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
        ``(true_A, fixed_estimated_paths, jump_estimated_paths, fixed_mise,
        jump_mise, fixed_execution_time, jump_execution_time)``.
    """
    num_individuals_values = np.asarray(list(num_individuals_list), dtype=int)
    if num_individuals_values.size == 0:
        raise ValueError("num_individuals must contain at least one value")

    true_A = np.stack(
        [generator(k / num_intervals) for k in range(num_intervals + 1)]
    )
    fixed_estimated_paths = []
    jump_estimated_paths = []
    fixed_mise_values = []
    jump_mise_values = []
    fixed_execution_times = []
    jump_execution_times = []

    fixed_emb = Fixed_Embedder(
        rank,
        num_intervals=num_intervals,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )
    jump_emb = Jump_Embedder(
        rank,
        num_intervals=num_intervals,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )

    for num_individuals in num_individuals_values:
        observations = Observation(generator, rank, fixed_times, num_intervals)
        fixed_paths = observations.simulate_fixed_observations(num_individuals, )
        jump_times, jump_states = observations.simulate_individual_jump_times_observations(num_individuals)

        start = perf_counter()
        fixed_estimated_A = fixed_emb.optimise_A(fixed_times, fixed_paths)
        fixed_execution_times.append(perf_counter() - start)

        start = perf_counter()
        jump_estimated_A = jump_emb.optimise_A(jump_times, jump_states)
        jump_execution_times.append(perf_counter() - start)

        fixed_estimated_paths.append(fixed_estimated_A)
        jump_estimated_paths.append(jump_estimated_A)
        fixed_mise_values.append(np.mean((true_A - fixed_estimated_A) ** 2))
        jump_mise_values.append(np.mean((true_A - jump_estimated_A) ** 2))

        print(
            f"n={num_individuals}: "
            f"fixed-time={fixed_execution_times[-1]:.3f}s, "
            f"jump-time={jump_execution_times[-1]:.3f}s"
        )

    fixed_mise_values = np.asarray(fixed_mise_values)
    jump_mise_values = np.asarray(jump_mise_values)
    fixed_execution_times = np.asarray(fixed_execution_times)
    jump_execution_times = np.asarray(jump_execution_times)

    save_results(
        results_file,
        num_individuals_values,
        fixed_mise_values,
        jump_mise_values,
        fixed_execution_times,
        jump_execution_times,
    )

    return (
        true_A,
        fixed_estimated_paths,
        jump_estimated_paths,
        fixed_mise_values,
        jump_mise_values,
        fixed_execution_times,
        jump_execution_times,
    )


def generate_data(
    results_file=DEFAULT_RESULTS_FILE,
    target_jumps=8,
    rank=2,
    num_intervals=1000,
    num_individuals_list=list(range(100, 10000, 100)),
    max_iter=1000,
):
    """Run the original fixed-time versus jump-time experiment and save it."""
    observations = Observation(trig_generator(), rank)
    observations.tune_generator_for_target_jumps(
        lambda A: trig_generator(A=A), target_jumps
    )

    # Retain this calculation from the original experiment so the tuned
    # generator's expected number of jumps can be inspected while debugging.
    observations.expected_jumps(np.full(rank, 1.0 / rank))

    times = np.linspace(0.0, 1.0, target_jumps)
    return run_experiments(
        observations.generator,
        rank,
        times,
        num_intervals,
        num_individuals_list,
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
    fixed_mise = data["fixed_mise"]
    jump_mise = data["jump_mise"]
    fixed_execution_time = data["fixed_execution_seconds"]
    jump_execution_time = data["jump_execution_seconds"]

    log_x = np.log(num_individuals)
    fixed_slope = linregress(log_x, np.log(fixed_mise)).slope
    jump_slope = linregress(log_x, np.log(jump_mise)).slope
    mise_summary = (
        f"Fixed slope = {fixed_slope:.3f}, "
        f"Jump slope = {jump_slope:.3f}"
    )
    print("mise summary:", mise_summary)

    mise_title = "log(MISE) vs. log(number of individuals)"

    fig, (mise_ax, time_ax) = plt.subplots(1, 2, figsize=(12, 5))

    mise_ax.loglog(
        num_individuals,
        fixed_mise,
        marker="o",
        color="tab:blue",
        label="fixed-time",
    )
    mise_ax.loglog(
        num_individuals,
        jump_mise,
        marker="o",
        color="tab:orange",
        label="jump-time",
    )
    mise_ax.set_xlabel("Number of individuals sampled", fontsize=LABEL_FONT_SIZE)
    mise_ax.set_ylabel("Mean integrated square error", fontsize=LABEL_FONT_SIZE)
    mise_ax.set_title(mise_title, fontsize=TITLE_FONT_SIZE)
    mise_ax.grid(True, which="both", alpha=0.3)
    mise_ax.legend(fontsize=LEGEND_FONT_SIZE)
    mise_ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    time_ax.plot(
        num_individuals,
        fixed_execution_time,
        marker="o",
        color="tab:blue",
        label="fixed-time",
    )
    time_ax.plot(
        num_individuals,
        jump_execution_time,
        marker="o",
        color="tab:orange",
        label="jump-time",
    )
    time_ax.set_xlabel("Number of individuals sampled", fontsize=LABEL_FONT_SIZE)
    time_ax.set_ylabel("Execution time (seconds)", fontsize=LABEL_FONT_SIZE)
    time_ax.set_title("Execution Time vs. Number of Individuals", fontsize=TITLE_FONT_SIZE)
    time_ax.grid(True, alpha=0.3)
    time_ax.legend(fontsize=LEGEND_FONT_SIZE)
    time_ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    fig.tight_layout()

    if figure_file is not None:
        figure_file = Path(figure_file)
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, (mise_ax, time_ax)

def main():
    I_1 = 1e0
    I_N = 1e6
    N = 6
    b = (np.log(I_N) - np.log(I_1)) / (N-1)
    a = np.log(I_1) - b
    i = a + b * np.arange(1,N+1)
    I = np.exp(i)
    print(I)

    parser = argparse.ArgumentParser(
        description="Compare fixed-time and jump-time embedding experiments."
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
        generate_data(args.results_file, num_individuals_list=I)
    plot_results_from_file(args.results_file, figure_file=args.figure_file)


if __name__ == "__main__":
    main()
