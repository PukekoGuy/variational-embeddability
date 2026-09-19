"""Check trial aggregation without running expensive optimisations."""
import csv
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TrialAveragesTest(unittest.TestCase):
    def test_average_errors_and_times_for_each_size(self):
        for name, methods in (
            ('fixed_vs_jump_time_fast', ('Fixed_Embedder', 'Jump_Embedder')),
            ('fixed_adjoint_vs_gradient_fast', ('adjoint_Embedder', 'gradient_Embedder')),
        ):
            with self.subTest(module=name), tempfile.TemporaryDirectory() as folder:
                module = importlib.import_module(name)
                clock = [0.0]

                def make_fit():
                    estimates = iter((1., 3., 2., 4.))

                    def fit(*args):
                        value = next(estimates)
                        clock[0] += value
                        return np.full((5, 2, 2), value)
                    return fit

                filename = Path(folder) / 'results.csv'
                with patch.object(module, 'perf_counter', side_effect=lambda: clock[0]), \
                     patch.object(getattr(module, methods[0]), 'optimise_A', side_effect=make_fit()), \
                     patch.object(getattr(module, methods[1]), 'optimise_A', side_effect=make_fit()):
                    result = module.run_experiments(
                        lambda t: np.zeros((2, 2)), 2, [0., 1.], 4, [3, 6],
                        results_file=filename, seed=7, num_trials=2,
                    )
                # Mean squared error is 5 and 10; squaring the mean would give 4 and 9.
                for values in result[3:5]:
                    np.testing.assert_allclose(values, [5., 10.])
                for values in result[5:7]:
                    np.testing.assert_allclose(values, [2., 3.])
                for paths in result[1:3]:
                    np.testing.assert_allclose(paths[0], 2.)
                    np.testing.assert_allclose(paths[1], 3.)
                with filename.open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 2)
                self.assertEqual([int(row['num_trials']) for row in rows], [2, 2])
                np.testing.assert_allclose([float(row[module.FIELDS[1]]) for row in rows], [5., 10.])

    def test_reject_invalid_trial_counts(self):
        for name in ('fixed_vs_jump_time_fast', 'fixed_adjoint_vs_gradient_fast'):
            module = importlib.import_module(name)
            for trials in (0, -1, 1.5, True):
                with self.subTest(module=name, trials=trials), self.assertRaises(ValueError):
                    module.run_experiments(
                        lambda t: np.zeros((2, 2)), 2, [0., 1.], 4, [3], num_trials=trials,
                    )


if __name__ == '__main__':
    unittest.main()
