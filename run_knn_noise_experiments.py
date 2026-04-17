from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csds452_project_spring_2026.knn_counterfactual import ( # type: ignore
    DEFAULT_DATA_PATTERN,
    aggregate_result_records,
    evaluate_replica_paths,
    iter_replica_paths,
    save_results,
)


CAUSAL_PREFIX_COLUMNS = 5
CONTINUOUS_FEATURE_INDICES = (0, 1, 2, 3, 4, 5)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "experiments" / "knn_counterfactual"


@dataclass(frozen=True)
class NoiseExperimentSpec:
    name: str
    description: str
    mode: str
    seed: int
    gaussian_mean: float = 0.0
    gaussian_std: float = 0.0
    num_drop_columns: int = 0
    drop_value: float = 0.0
    drop_scope: str = "global"
    continuous_feature_indices: tuple[int, ...] = CONTINUOUS_FEATURE_INDICES


def build_experiment_specs() -> list[NoiseExperimentSpec]:
    return [
        NoiseExperimentSpec(
            name="gaussian_continuous_std_0p10",
            description="Gaussian noise on continuous covariates only.",
            mode="gaussian",
            seed=20260415,
            gaussian_std=0.10,
        ),
        NoiseExperimentSpec(
            name="drop_global_5cols",
            description="Drop the same five covariate columns across all replicas.",
            mode="drop",
            seed=20260416,
            num_drop_columns=5,
            drop_scope="global",
        ),
        NoiseExperimentSpec(
            name="both_per_replication_std_0p10_drop5",
            description=(
                "Per-replica column drop followed by Gaussian noise on visible continuous covariates."
            ),
            mode="both",
            seed=20260417,
            gaussian_std=0.10,
            num_drop_columns=5,
            drop_scope="per_replication",
        ),
    ]


def choose_drop_columns(
    rng: np.random.Generator, n_features: int, num_drop_columns: int
) -> list[int]:
    if num_drop_columns < 0 or num_drop_columns > n_features:
        raise ValueError(
            f"num_drop_columns must be between 0 and {n_features}. Received {num_drop_columns}."
        )
    if num_drop_columns == 0:
        return []
    return np.sort(rng.choice(n_features, size=num_drop_columns, replace=False)).tolist()


def apply_gaussian_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    feature_indices: tuple[int, ...],
    mean: float,
    std: float,
    protected_columns: list[int] | None = None,
) -> np.ndarray:
    noisy_x = np.array(x, copy=True)
    if std < 0:
        raise ValueError("gaussian_std must be non-negative.")

    protected = set(protected_columns or [])
    active_columns = [index for index in feature_indices if index not in protected]
    if not active_columns:
        return noisy_x

    noise = rng.normal(loc=mean, scale=std, size=(noisy_x.shape[0], len(active_columns)))
    noisy_x[:, active_columns] = noisy_x[:, active_columns] + noise
    return noisy_x


def apply_feature_drop(x: np.ndarray, dropped_columns: list[int], drop_value: float) -> np.ndarray:
    dropped_x = np.array(x, copy=True)
    if dropped_columns:
        dropped_x[:, dropped_columns] = drop_value
    return dropped_x


def apply_noise_experiment(
    data: np.ndarray,
    spec: NoiseExperimentSpec,
    rng: np.random.Generator,
    dropped_columns: list[int],
) -> np.ndarray:
    noisy_data = np.array(data, copy=True)
    x = noisy_data[:, CAUSAL_PREFIX_COLUMNS:]

    if spec.mode in {"drop", "both"}:
        x = apply_feature_drop(x, dropped_columns=dropped_columns, drop_value=spec.drop_value)
    if spec.mode in {"gaussian", "both"}:
        x = apply_gaussian_noise(
            x=x,
            rng=rng,
            feature_indices=spec.continuous_feature_indices,
            mean=spec.gaussian_mean,
            std=spec.gaussian_std,
            protected_columns=dropped_columns,
        )

    noisy_data[:, CAUSAL_PREFIX_COLUMNS:] = x
    return noisy_data


def save_replica_csv(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", fmt="%.16g")


def materialize_noisy_experiment(
    source_paths: list[Path],
    spec: NoiseExperimentSpec,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    noisy_paths: list[Path] = []
    replica_metadata: list[dict[str, Any]] = []
    global_dropped_columns: list[int] | None = None

    if spec.mode in {"drop", "both"} and spec.drop_scope == "global":
        first_data = np.loadtxt(source_paths[0], delimiter=",", dtype=float)
        n_features = first_data.shape[1] - CAUSAL_PREFIX_COLUMNS
        global_dropped_columns = choose_drop_columns(rng, n_features, spec.num_drop_columns)

    for source_path in source_paths:
        data = np.loadtxt(source_path, delimiter=",", dtype=float)
        n_features = data.shape[1] - CAUSAL_PREFIX_COLUMNS
        if spec.mode in {"drop", "both"}:
            if spec.drop_scope == "global":
                dropped_columns = list(global_dropped_columns or [])
            else:
                dropped_columns = choose_drop_columns(rng, n_features, spec.num_drop_columns)
        else:
            dropped_columns = []

        noisy_data = apply_noise_experiment(data=data, spec=spec, rng=rng, dropped_columns=dropped_columns)
        output_path = dataset_dir / source_path.name
        save_replica_csv(output_path, noisy_data)
        noisy_paths.append(output_path)
        replica_metadata.append(
            {
                "source_path": str(source_path.resolve()),
                "output_path": str(output_path.resolve()),
                "dropped_columns": dropped_columns,
            }
        )

    experiment_metadata = {
        "experiment": asdict(spec),
        "replicas": replica_metadata,
        "global_dropped_columns": global_dropped_columns,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return noisy_paths, experiment_metadata


def write_comparison_table(
    comparison_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "comparison.csv"
    fieldnames = [
        "experiment_name",
        "dataset_type",
        "counterfactual_rmse",
        "control_counterfactual_rmse",
        "treated_counterfactual_rmse",
        "pehe",
        "ate_hat",
        "ate_true",
        "ate_abs_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return {"json": json_path, "csv": csv_path}


def run_experiment_suite(
    input_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pattern: str = DEFAULT_DATA_PATTERN,
    k: int = 5,
    metric: str = "euclidean",
    scale: bool = True,
    weighted: bool = False,
) -> dict[str, Any]:
    source_paths = iter_replica_paths(input_dir=input_dir, pattern=pattern)
    if not source_paths:
        raise ValueError("No IHDP replica CSVs were found for the experiment suite.")

    base_output_dir = Path(output_dir)
    original_results = evaluate_replica_paths(
        paths=source_paths,
        k=k,
        metric=metric,
        scale=scale,
        weighted=weighted,
    )
    original_dir = base_output_dir / "original" / "results"
    save_results(
        output_dir=original_dir,
        results=original_results,
        extra_metadata={
            "dataset_type": "original",
            "input_dir": str(Path(source_paths[0]).parent.resolve()),
            "pattern": pattern,
            "k": k,
            "metric": metric,
            "scale": scale,
            "weighted": weighted,
        },
    )

    comparison_rows: list[dict[str, Any]] = []
    original_aggregate = aggregate_result_records([{
        "counterfactual_rmse": row.counterfactual_rmse,
        "control_counterfactual_rmse": row.control_counterfactual_rmse,
        "treated_counterfactual_rmse": row.treated_counterfactual_rmse,
        "pehe": row.pehe,
        "ate_hat": row.ate_hat,
        "ate_true": row.ate_true,
        "ate_abs_error": row.ate_abs_error,
    } for row in original_results])
    if original_aggregate is not None:
        comparison_rows.append(
            {"experiment_name": "original", "dataset_type": "original", **original_aggregate}
        )

    noisy_runs: dict[str, dict[str, Any]] = {}
    for spec in build_experiment_specs():
        experiment_root = base_output_dir / "noisy" / spec.name
        noisy_paths, experiment_metadata = materialize_noisy_experiment(
            source_paths=source_paths,
            spec=spec,
            output_dir=experiment_root,
        )
        noisy_results = evaluate_replica_paths(
            paths=noisy_paths,
            k=k,
            metric=metric,
            scale=scale,
            weighted=weighted,
        )
        result_paths = save_results(
            output_dir=experiment_root / "results",
            results=noisy_results,
            extra_metadata={
                "dataset_type": "noisy",
                "experiment_name": spec.name,
                "k": k,
                "metric": metric,
                "scale": scale,
                "weighted": weighted,
                "noise_spec": asdict(spec),
            },
        )
        aggregate = aggregate_result_records([{
            "counterfactual_rmse": row.counterfactual_rmse,
            "control_counterfactual_rmse": row.control_counterfactual_rmse,
            "treated_counterfactual_rmse": row.treated_counterfactual_rmse,
            "pehe": row.pehe,
            "ate_hat": row.ate_hat,
            "ate_true": row.ate_true,
            "ate_abs_error": row.ate_abs_error,
        } for row in noisy_results])
        if aggregate is not None:
            comparison_rows.append(
                {"experiment_name": spec.name, "dataset_type": "noisy", **aggregate}
            )
        noisy_runs[spec.name] = {
            "dataset_dir": experiment_root / "datasets",
            "result_dir": experiment_root / "results",
            "config": experiment_metadata,
            "result_paths": result_paths,
        }

    comparison_paths = write_comparison_table(
        comparison_rows=comparison_rows,
        output_dir=base_output_dir / "comparison",
    )
    return {
        "source_paths": source_paths,
        "original_result_dir": original_dir,
        "noisy_runs": noisy_runs,
        "comparison_paths": comparison_paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate noisy IHDP CSV experiment suites, run k-NN counterfactual estimation, "
            "and save results for original and noisy data."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing IHDP replica CSVs. Defaults to the bundled dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store original/noisy datasets and experiment results.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_DATA_PATTERN,
        help="Filename glob for replica CSVs. Default: ihdp_npci_*.csv.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN.")
    parser.add_argument(
        "--metric",
        choices=("euclidean", "manhattan"),
        default="euclidean",
        help="Distance metric for k-NN.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable z-score scaling before k-NN distance computation.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use inverse-distance weighting in k-NN.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_experiment_suite(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        k=args.k,
        metric=args.metric,
        scale=not args.no_scale,
        weighted=args.weighted,
    )


if __name__ == "__main__":
    main()
