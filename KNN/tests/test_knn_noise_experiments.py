import tempfile
import unittest
from pathlib import Path

import numpy as np

from csds452_project_spring_2026.KNN.run_knn_noise_experiments import (
    NoiseExperimentSpec,
    apply_noise_experiment,
    run_experiment_suite,
)


def _synthetic_replica_matrix(offset: float) -> np.ndarray:
    treatment = np.array([0.0, 1.0, 0.0, 1.0])[:, None]
    y_factual = np.array([1.0, 2.0, 3.0, 4.0])[:, None] + offset
    y_cfactual = np.array([2.0, 1.0, 4.0, 3.0])[:, None] + offset
    mu0 = np.array([1.0, 1.5, 3.0, 3.5])[:, None] + offset
    mu1 = np.array([2.0, 2.5, 4.0, 4.5])[:, None] + offset
    continuous = np.tile(np.arange(6, dtype=float), (4, 1)) + offset
    binary = np.tile(np.array([0.0, 1.0] * 9 + [0.0]), (4, 1))
    x = np.concatenate([continuous, binary], axis=1)
    return np.concatenate([treatment, y_factual, y_cfactual, mu0, mu1, x], axis=1)


class KNNNoiseExperimentTests(unittest.TestCase):
    def test_apply_noise_experiment_keeps_causal_prefix_intact(self) -> None:
        data = _synthetic_replica_matrix(offset=0.0)
        spec = NoiseExperimentSpec(
            name="combined",
            description="test",
            mode="both",
            seed=1,
            gaussian_std=0.1,
            num_drop_columns=2,
        )
        rng = np.random.default_rng(1)
        noisy = apply_noise_experiment(data, spec, rng)

        np.testing.assert_allclose(noisy[:, :5], data[:, :5])
        self.assertEqual(noisy.shape[1], data.shape[1] - 2)
        self.assertFalse(np.array_equal(noisy[:, 5:9], data[:, 5:9]))

    def test_run_experiment_suite_creates_original_and_noisy_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            input_dir.mkdir()
            np.savetxt(input_dir / "ihdp_npci_1.csv", _synthetic_replica_matrix(0.0), delimiter=",", fmt="%.6f")
            np.savetxt(input_dir / "ihdp_npci_2.csv", _synthetic_replica_matrix(1.0), delimiter=",", fmt="%.6f")

            output_dir = root / "outputs"
            run_experiment_suite(input_dir=input_dir, output_dir=output_dir)

            self.assertTrue((output_dir / "original" / "results" / "summary.txt").exists())
            self.assertTrue((output_dir / "comparison" / "comparison.csv").exists())
            self.assertTrue(
                (
                    output_dir
                    / "noisy"
                    / "gaussian_continuous_std_0p10"
                    / "results"
                    / "metrics.json"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "noisy"
                    / "drop_5cols"
                    / "datasets"
                    / "ihdp_npci_1.csv"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
