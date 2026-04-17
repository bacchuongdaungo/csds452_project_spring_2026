from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csds452_project_spring_2026.KNN.knn_counterfactual import (  # type: ignore
    DEFAULT_DATA_PATTERN,
    aggregate_result_records,
    evaluate_replica_paths,
    iter_replica_paths,
    save_results,
)
from csds452_project_spring_2026.noisify_ihdp import (  # type: ignore
    CAUSAL_PREFIX_COLUMNS,
    apply_combined_noise,
    apply_gaussian_noise,
    drop_feature_columns,
    run_noise_pipeline,
)


CONTINUOUS_FEATURE_INDICES = (0, 1, 2, 3, 4, 5)
DEFAULT_OUTPUT_DIR = HERE / "experiments" / "knn_counterfactual"


@dataclass(frozen=True)
class NoiseExperimentSpec:
    name: str
    description: str
    mode: str
    seed: int
    gaussian_mean: float = 0.0
    gaussian_std: float = 0.0
    num_drop_columns: int = 0
    continuous_feature_indices: tuple[int, ...] = CONTINUOUS_FEATURE_INDICES


def build_experiment_specs() -> list[NoiseExperimentSpec]:
    return [
        NoiseExperimentSpec(
            name="gaussian_continuous_std_0p10",
            description="Gaussian noise on selected continuous covariates only.",
            mode="gaussian",
            seed=20260415,
            gaussian_std=0.10,
        ),
        NoiseExperimentSpec(
            name="drop_5cols",
            description="Remove five covariate columns from each replica.",
            mode="drop",
            seed=20260416,
            num_drop_columns=5,
        ),
        NoiseExperimentSpec(
            name="both_std_0p10_drop5",
            description="Remove five covariate columns, then add Gaussian noise to the remaining selected covariates.",
            mode="both",
            seed=20260417,
            gaussian_std=0.10,
            num_drop_columns=5,
        ),
    ]


def apply_noise_experiment(
    data: np.ndarray,
    spec: NoiseExperimentSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the current CSV-based noise workflow to one in-memory replica."""
    noisy_prefix = np.array(data[:, :CAUSAL_PREFIX_COLUMNS], copy=True)
    x = np.array(data[:, CAUSAL_PREFIX_COLUMNS:], copy=True)

    if spec.mode == "gaussian":
        noisy_x = apply_gaussian_noise(
            x=x,
            rng=rng,
            gaussian_mean=spec.gaussian_mean,
            gaussian_std=spec.gaussian_std,
            continuous_feature_indices=list(spec.continuous_feature_indices),
        )
    elif spec.mode == "drop":
        noisy_x, _, _ = drop_feature_columns(
            x=x,
            rng=rng,
            num_drop_columns=spec.num_drop_columns,
        )
    elif spec.mode == "both":
        noisy_x, _, _ = apply_combined_noise(
            x=x,
            rng=rng,
            gaussian_mean=spec.gaussian_mean,
            gaussian_std=spec.gaussian_std,
            num_drop_columns=spec.num_drop_columns,
            continuous_feature_indices=list(spec.continuous_feature_indices),
        )
    else:
        raise ValueError(f"Unsupported mode: {spec.mode}")

    return np.concatenate([noisy_prefix, noisy_x], axis=1)


def materialize_noisy_experiment(
    source_paths: list[Path],
    spec: NoiseExperimentSpec,
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    noisy_paths: list[Path] = []
    replica_metadata: list[dict[str, Any]] = []

    for replica_index, source_path in enumerate(source_paths):
        output_path = dataset_dir / source_path.name
        replica_seed = spec.seed + replica_index
        result = run_noise_pipeline(
            input_path=source_path,
            output_path=output_path,
            mode=spec.mode,
            seed=replica_seed,
            gaussian_mean=spec.gaussian_mean,
            gaussian_std=spec.gaussian_std if spec.mode in {"gaussian", "both"} else None,
            num_drop_columns=spec.num_drop_columns if spec.mode in {"drop", "both"} else None,
            continuous_feature_indices=spec.continuous_feature_indices,
            save_mask=spec.mode in {"drop", "both"},
            verbose=False,
        )
        noisy_paths.append(Path(result["output_path"]))
        replica_metadata.append(
            {
                "source_path": str(Path(source_path).resolve()),
                "output_path": str(Path(result["output_path"]).resolve()),
                "seed": replica_seed,
                "metadata_path": str(Path(result["metadata_path"]).resolve()),
                "dropped_columns": result["metadata"]["dropped_column_indices"],
                "mask_path": result["metadata"]["mask_path"],
                "x_shape_before": result["metadata"]["shapes"]["x_before"],
                "x_shape_after": result["metadata"]["shapes"]["x_after"],
            }
        )

    experiment_metadata = {
        "experiment": asdict(spec),
        "replicas": replica_metadata,
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
    original_aggregate = aggregate_result_records(
        [
            {
                "counterfactual_rmse": row.counterfactual_rmse,
                "control_counterfactual_rmse": row.control_counterfactual_rmse,
                "treated_counterfactual_rmse": row.treated_counterfactual_rmse,
                "pehe": row.pehe,
                "ate_hat": row.ate_hat,
                "ate_true": row.ate_true,
                "ate_abs_error": row.ate_abs_error,
            }
            for row in original_results
        ]
    )
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
        aggregate = aggregate_result_records(
            [
                {
                    "counterfactual_rmse": row.counterfactual_rmse,
                    "control_counterfactual_rmse": row.control_counterfactual_rmse,
                    "treated_counterfactual_rmse": row.treated_counterfactual_rmse,
                    "pehe": row.pehe,
                    "ate_hat": row.ate_hat,
                    "ate_true": row.ate_true,
                    "ate_abs_error": row.ate_abs_error,
                }
                for row in noisy_results
            ]
        )
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
