from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DATA_GLOB = "data/ihdp_dataset/csv/ihdp_npci_*.csv"


@dataclass(frozen=True)
class IHDPDataset:
    path: Path
    treatment: np.ndarray
    y_factual: np.ndarray
    y_cfactual: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    x: np.ndarray

    @property
    def true_y0(self) -> np.ndarray:
        return np.where(self.treatment == 0, self.y_factual, self.y_cfactual)

    @property
    def true_y1(self) -> np.ndarray:
        return np.where(self.treatment == 1, self.y_factual, self.y_cfactual)


@dataclass(frozen=True)
class KNNCounterfactualResult:
    dataset: Path
    k: int
    metric: str
    scaled: bool
    weighted: bool
    y_cfactual_hat: np.ndarray
    y0_hat: np.ndarray
    y1_hat: np.ndarray
    ite_hat: np.ndarray
    counterfactual_rmse: float
    control_counterfactual_rmse: float
    treated_counterfactual_rmse: float
    pehe: float
    ate_hat: float
    ate_true: float
    ate_abs_error: float


def load_ihdp_replica(path: str | Path) -> IHDPDataset:
    csv_path = Path(path)
    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"Expected IHDP matrix with >= 6 columns, got shape {data.shape}")

    treatment = data[:, 0].astype(int)
    return IHDPDataset(
        path=csv_path,
        treatment=treatment,
        y_factual=data[:, 1],
        y_cfactual=data[:, 2],
        mu0=data[:, 3],
        mu1=data[:, 4],
        x=data[:, 5:],
    )


def standardize_features(x: np.ndarray) -> np.ndarray:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)
    return (x - means) / stds


def pairwise_distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.sqrt(squared)
    if metric == "manhattan":
        return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    raise ValueError(f"Unsupported metric: {metric}")


def _predict_from_reference(
    x_query: np.ndarray,
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    k: int,
    metric: str,
    weighted: bool,
) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1")
    if x_reference.shape[0] == 0:
        raise ValueError("Opposite treatment group is empty; cannot impute counterfactuals")

    effective_k = min(k, x_reference.shape[0])
    distances = pairwise_distances(x_query, x_reference, metric)
    neighbor_idx = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
    neighbor_distances = np.take_along_axis(distances, neighbor_idx, axis=1)

    # Sort the selected neighbors to make the returned set deterministic.
    order = np.argsort(neighbor_distances, axis=1)
    neighbor_idx = np.take_along_axis(neighbor_idx, order, axis=1)
    neighbor_distances = np.take_along_axis(neighbor_distances, order, axis=1)

    neighbor_outcomes = y_reference[neighbor_idx]
    if not weighted:
        return neighbor_outcomes.mean(axis=1)

    weights = 1.0 / np.maximum(neighbor_distances, 1e-12)
    return np.sum(weights * neighbor_outcomes, axis=1) / np.sum(weights, axis=1)


def estimate_counterfactuals(
    dataset: IHDPDataset,
    k: int = 5,
    metric: str = "euclidean",
    scale: bool = True,
    weighted: bool = False,
) -> KNNCounterfactualResult:
    x = standardize_features(dataset.x) if scale else dataset.x.copy()
    treated_mask = dataset.treatment == 1
    control_mask = ~treated_mask

    x_control = x[control_mask]
    x_treated = x[treated_mask]
    y_control = dataset.y_factual[control_mask]
    y_treated = dataset.y_factual[treated_mask]

    control_cf = _predict_from_reference(
        x_query=x_control,
        x_reference=x_treated,
        y_reference=y_treated,
        k=k,
        metric=metric,
        weighted=weighted,
    )
    treated_cf = _predict_from_reference(
        x_query=x_treated,
        x_reference=x_control,
        y_reference=y_control,
        k=k,
        metric=metric,
        weighted=weighted,
    )

    y_cfactual_hat = np.empty_like(dataset.y_cfactual)
    y_cfactual_hat[control_mask] = control_cf
    y_cfactual_hat[treated_mask] = treated_cf

    y0_hat = np.where(control_mask, dataset.y_factual, y_cfactual_hat)
    y1_hat = np.where(treated_mask, dataset.y_factual, y_cfactual_hat)
    ite_hat = y1_hat - y0_hat
    true_ite = dataset.mu1 - dataset.mu0

    control_rmse = rmse(control_cf, dataset.y_cfactual[control_mask])
    treated_rmse = rmse(treated_cf, dataset.y_cfactual[treated_mask])

    return KNNCounterfactualResult(
        dataset=dataset.path,
        k=k,
        metric=metric,
        scaled=scale,
        weighted=weighted,
        y_cfactual_hat=y_cfactual_hat,
        y0_hat=y0_hat,
        y1_hat=y1_hat,
        ite_hat=ite_hat,
        counterfactual_rmse=rmse(y_cfactual_hat, dataset.y_cfactual),
        control_counterfactual_rmse=control_rmse,
        treated_counterfactual_rmse=treated_rmse,
        pehe=rmse(ite_hat, true_ite),
        ate_hat=float(np.mean(ite_hat)),
        ate_true=float(np.mean(true_ite)),
        ate_abs_error=float(abs(np.mean(ite_hat) - np.mean(true_ite))),
    )


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def iter_replica_paths(input_path: str | None) -> list[Path]:
    if input_path:
        return [Path(input_path)]

    base_dir = Path(__file__).resolve().parent
    return sorted(base_dir.glob(DEFAULT_DATA_GLOB))


def summarize_results(results: Iterable[KNNCounterfactualResult]) -> str:
    rows = list(results)
    if not rows:
        return "No IHDP replicas were found."

    header = (
        "dataset".ljust(18)
        + "k".rjust(4)
        + "  cf_rmse".rjust(10)
        + "  ctrl_rmse".rjust(11)
        + "  trt_rmse".rjust(10)
        + "  pehe".rjust(8)
        + "  ate_hat".rjust(10)
        + "  ate_true".rjust(10)
        + "  ate_abs".rjust(10)
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            row.dataset.name.ljust(18)
            + f"{row.k:>4}"
            + f"{row.counterfactual_rmse:>10.4f}"
            + f"{row.control_counterfactual_rmse:>11.4f}"
            + f"{row.treated_counterfactual_rmse:>10.4f}"
            + f"{row.pehe:>8.4f}"
            + f"{row.ate_hat:>10.4f}"
            + f"{row.ate_true:>10.4f}"
            + f"{row.ate_abs_error:>10.4f}"
        )

    if len(rows) > 1:
        avg = lambda attr: float(np.mean([getattr(row, attr) for row in rows]))
        lines.append("-" * len(header))
        lines.append(
            "AVERAGE".ljust(18)
            + f"{rows[0].k:>4}"
            + f"{avg('counterfactual_rmse'):>10.4f}"
            + f"{avg('control_counterfactual_rmse'):>11.4f}"
            + f"{avg('treated_counterfactual_rmse'):>10.4f}"
            + f"{avg('pehe'):>8.4f}"
            + f"{avg('ate_hat'):>10.4f}"
            + f"{avg('ate_true'):>10.4f}"
            + f"{avg('ate_abs_error'):>10.4f}"
        )

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate IHDP counterfactuals with opposite-group k-nearest neighbors."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to one IHDP replica CSV. Omit to evaluate all bundled replicas.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of opposite-group neighbors.")
    parser.add_argument(
        "--metric",
        choices=("euclidean", "manhattan"),
        default="euclidean",
        help="Distance metric over covariates.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use inverse-distance weighting instead of a plain mean over neighbors.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable z-score scaling of covariates before computing distances.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = iter_replica_paths(args.input)
    if not paths:
        raise SystemExit("No IHDP replica CSVs found.")

    results = [
        estimate_counterfactuals(
            dataset=load_ihdp_replica(path),
            k=args.k,
            metric=args.metric,
            scale=not args.no_scale,
            weighted=args.weighted,
        )
        for path in paths
    ]
    print(summarize_results(results))


if __name__ == "__main__":
    main()
