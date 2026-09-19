"""Compare fixed-time and jump-time generator estimates in a 2-by-2 figure."""

import matplotlib.pyplot as plt
import numpy as np

from cosmetics import PLOT_FONT_SIZE
from experiment import plot_2D_generator_path, trig_generator
from individual_data_adjoint import run_experiments as run_fixed_experiments
from individual_jumps_gradient import run_experiments as run_jump_experiments


def plot_comparison(
    num_individuals_range,
    num_observations=(20, 10),
    constant=10,
    *,
    num_intervals=1000,
    mu=1e-2,
    kappa=1.0,
    max_iter=500,
    plot_frequency=1,
    fontsize=PLOT_FONT_SIZE,
    figsize=(16, 12),
    show=True,
):
    """Simulate, fit, and plot two-state generator paths for both methods.

    ``num_individuals_range`` supplies the shared panel sizes for all four
    plots (values are converted to integers, as in ``generate_data``).
    ``num_observations`` contains the top and bottom row's observation counts.
    For fixed-time data, these count equally spaced observations including
    both endpoints of [0, 1]; the generator targets count / ``constant``
    expected jumps. For jump-time data, the generator targets count expected
    jumps, counting each jump as an observation, excluding the initial state.

    Each panel uses its own tuned trigonometric generator and true path.
    Returns ``(fig, axes)`` with ``axes.shape == (2, 2)``. Use ``show=False``
    to customize or save the figure before displaying it.

    Example::

        fig, axes = plot_comparison(
            range(100, 501, 100), num_observations=(4, 8), constant=2,
        )
    """
    panel_sizes = np.asarray(list(num_individuals_range), dtype=int)
    if panel_sizes.ndim != 1 or panel_sizes.size == 0 or np.any(panel_sizes < 1):
        raise ValueError("num_individuals_range must contain positive panel sizes")
    observation_counts = tuple(num_observations)
    if len(observation_counts) != 2 or any(
        not isinstance(count, (int, np.integer)) or count < 2
        for count in observation_counts
    ):
        raise ValueError("num_observations must contain two integers of at least 2")
    if not np.isfinite(constant) or constant <= 0:
        raise ValueError("constant must be finite and positive")
    if not isinstance(num_intervals, (int, np.integer)) or num_intervals < 1:
        raise ValueError("num_intervals must be a positive integer")
    if max(observation_counts) > num_intervals + 1:
        raise ValueError("num_intervals is too small for the observation counts")
    if not isinstance(plot_frequency, (int, np.integer)) or plot_frequency < 1:
        raise ValueError("plot_frequency must be a positive integer")

    common = dict(
        rank=2,
        num_intervals=num_intervals,
        num_individuals_range=panel_sizes,
        mu=mu,
        kappa=kappa,
        max_iter=max_iter,
    )
    results = []
    for count in observation_counts:
        fixed = run_fixed_experiments(
            trig_generator(),
            target_num_jumps=count / constant,
            num_observations=count,
            **common,
        )
        jump = run_jump_experiments(
            trig_generator(), target_num_jumps=count, **common,
        )
        results.append((fixed, jump))

    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    for row, (count, (fixed, jump)) in enumerate(zip(observation_counts, results)):
        titles = (
            f"Fixed-time adjoint: {count} observations\n"
            f"Expected jumps = {count / constant:g}",
            f"Jump-time gradient: {count} expected observations\n"
            f"Expected jumps = {count}",
        )
        for col, (estimated_paths, true_path, _) in enumerate((fixed, jump)):
            plot_2D_generator_path(
                estimated_paths,
                true_path,
                panel_sizes,
                times=np.linspace(0.0, 1.0, count) if col == 0 else None,
                plot_frequency=plot_frequency,
                ax=axes[row, col],
                title=titles[col],
                fontsize=fontsize,
            )

    if show:
        plt.show()
    return fig, axes

def main():
    medium_range = np.linspace(500,5000,10)
    plot_comparison(num_individuals_range=medium_range, num_observations=(20, 10))

if __name__ == "__main__":
    main()