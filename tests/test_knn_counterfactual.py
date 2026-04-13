import unittest
from pathlib import Path

import numpy as np

from csds452_project_spring_2026.knn_counterfactual import (
    IHDPDataset,
    estimate_counterfactuals,
    iter_replica_paths,
    load_ihdp_replica,
)


class KNNCounterfactualTests(unittest.TestCase):
    def test_load_ihdp_replica_shape(self) -> None:
        dataset = load_ihdp_replica(
            Path("csds452_project_spring_2026/data/ihdp_dataset/csv/ihdp_npci_1.csv")
        )
        self.assertEqual(dataset.x.shape, (747, 25))
        self.assertEqual(dataset.treatment.shape, (747,))

    def test_exact_match_counterfactuals(self) -> None:
        dataset = IHDPDataset(
            path=Path("synthetic.csv"),
            treatment=np.array([0, 0, 1, 1]),
            y_factual=np.array([1.0, 2.0, 11.0, 12.0]),
            y_cfactual=np.array([11.0, 12.0, 1.0, 2.0]),
            mu0=np.array([1.0, 2.0, 1.0, 2.0]),
            mu1=np.array([11.0, 12.0, 11.0, 12.0]),
            x=np.array([[0.0], [1.0], [0.0], [1.0]]),
        )

        result = estimate_counterfactuals(dataset, k=1, scale=False)

        np.testing.assert_allclose(result.y_cfactual_hat, dataset.y_cfactual)
        np.testing.assert_allclose(result.ite_hat, np.array([10.0, 10.0, 10.0, 10.0]))
        self.assertAlmostEqual(result.counterfactual_rmse, 0.0)
        self.assertAlmostEqual(result.pehe, 0.0)

    def test_default_replica_glob_finds_bundled_csvs(self) -> None:
        self.assertEqual(len(iter_replica_paths(None)), 10)


if __name__ == "__main__":
    unittest.main()
