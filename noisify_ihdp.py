"""
IHDP covariate noise injection for robust causal learning experiments.

Usage examples:
1. Gaussian only
   python noisify_ihdp.py --input_path ihdp_npci_1.csv --output_path ihdp_gaussian.csv --mode gaussian --gaussian_std 0.1 --seed 42

2. Feature drop only
   python noisify_ihdp.py --input_path ihdp_npci_1.csv --output_path ihdp_drop.csv --mode drop --num_drop_columns 5 --seed 42

3. Combined mode
   python noisify_ihdp.py --input_path ihdp_npci_1.csv --output_path ihdp_both.csv --mode both --gaussian_std 0.1 --num_drop_columns 5 --continuous_feature_indices 0,1,2,3,4,5 --seed 42 --save_mask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CAUSAL_PREFIX_COLUMNS = 5
FEATURE_DROP_MASK_KEY = "feature_drop_mask"
MODE_CHOICES = ("gaussian", "drop", "both")


def parse_index_list(raw: str | Sequence[int] | None) -> list[int] | None:
    """Parse comma-separated feature indices into a sorted unique list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
        try:
            indices = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError("Feature indices must be a comma-separated list of integers.") from exc
    else:
        try:
            indices = [int(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature indices must be integers.") from exc

    if any(index < 0 for index in indices):
        raise ValueError("Feature indices must be non-negative.")
    return sorted(set(indices))


def load_ihdp_csv(input_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one IHDP CSV replica and return its full matrix and covariate block."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise ValueError(f"Input file does not exist: {input_file}")
    if input_file.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a .csv file: {input_file}")

    data = np.loadtxt(input_file, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.ndim != 2 or data.shape[1] < CAUSAL_PREFIX_COLUMNS + 1:
        raise ValueError(
            "Expected an IHDP CSV matrix with at least 6 columns "
            f"(5 causal columns + covariates). Found shape {data.shape}."
        )

    x = data[:, CAUSAL_PREFIX_COLUMNS:]
    return data, x


def validate_feature_indices(
    n_features: int,
    continuous_feature_indices: list[int] | None,
) -> None:
    """Validate explicit Gaussian feature indices."""
    if continuous_feature_indices is None:
        return

    out_of_range = [index for index in continuous_feature_indices if index >= n_features]
    if out_of_range:
        raise ValueError(
            f"continuous_feature_indices contains out-of-range indices {out_of_range}. "
            f"Valid feature indices are 0 to {n_features - 1}."
        )


def apply_gaussian_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    gaussian_mean: float,
    gaussian_std: float,
    continuous_feature_indices: list[int] | None = None,
) -> np.ndarray:
    """Add Gaussian noise to selected covariate columns."""
    if gaussian_std < 0:
        raise ValueError("gaussian_std must be non-negative.")

    x_noisy = np.array(x, dtype=np.float64, copy=True)
    if continuous_feature_indices is None:
        columns = np.arange(x_noisy.shape[1])
    else:
        columns = np.array(continuous_feature_indices, dtype=int)

    if columns.size == 0:
        return x_noisy

    noise = rng.normal(
        loc=gaussian_mean,
        scale=gaussian_std,
        size=(x_noisy.shape[0], columns.size),
    )
    x_noisy[:, columns] = x_noisy[:, columns] + noise
    return x_noisy


def drop_feature_columns(
    x: np.ndarray,
    rng: np.random.Generator,
    num_drop_columns: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Remove a random subset of covariate columns from the matrix."""
    n_features = x.shape[1]
    if num_drop_columns < 0 or num_drop_columns > n_features:
        raise ValueError(
            f"num_drop_columns must be between 0 and {n_features}. "
            f"Received {num_drop_columns}."
        )

    drop_mask = np.zeros(n_features, dtype=bool)
    if num_drop_columns == 0:
        return np.array(x, dtype=np.float64, copy=True), drop_mask, []

    dropped_columns = np.sort(rng.choice(n_features, size=num_drop_columns, replace=False)).tolist()
    drop_mask[dropped_columns] = True
    x_dropped = np.array(x[:, ~drop_mask], dtype=np.float64, copy=True)
    return x_dropped, drop_mask, dropped_columns


def remap_feature_indices_after_drop(
    feature_indices: list[int] | None,
    drop_mask: np.ndarray,
) -> list[int] | None:
    """Map original feature indices onto the compacted post-drop matrix."""
    if feature_indices is None:
        return None

    remapped: list[int] = []
    for index in feature_indices:
        if drop_mask[index]:
            continue
        remapped.append(index - int(np.sum(drop_mask[:index])))
    return remapped


def apply_combined_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    gaussian_mean: float,
    gaussian_std: float,
    num_drop_columns: int,
    continuous_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Apply feature drop first, then Gaussian noise to the remaining covariates."""
    x_dropped, drop_mask, dropped_columns = drop_feature_columns(
        x=x,
        rng=rng,
        num_drop_columns=num_drop_columns,
    )
    remapped_continuous_indices = remap_feature_indices_after_drop(
        continuous_feature_indices,
        drop_mask,
    )
    x_noisy = apply_gaussian_noise(
        x=x_dropped,
        rng=rng,
        gaussian_mean=gaussian_mean,
        gaussian_std=gaussian_std,
        continuous_feature_indices=remapped_continuous_indices,
    )
    return x_noisy, drop_mask, dropped_columns


def build_metadata(
    *,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    seed: int,
    gaussian_mean: float,
    gaussian_std: float | None,
    num_drop_columns: int | None,
    dropped_column_indices: list[int],
    continuous_feature_indices: list[int] | None,
    gaussian_feature_indices_used: list[int] | None,
    x_shape_before: Sequence[int],
    x_shape_after: Sequence[int],
    save_mask: bool,
    mask_shape: Sequence[int] | None,
) -> dict[str, Any]:
    """Create a JSON-serializable description of the perturbation run."""
    output_file = Path(output_path)
    mask_path = (
        output_file.with_name(f"{output_file.stem}_{FEATURE_DROP_MASK_KEY}.csv").resolve()
        if save_mask
        else None
    )
    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(output_file.resolve()),
        "causal_prefix_columns": CAUSAL_PREFIX_COLUMNS,
        "mode": mode,
        "seed": int(seed),
        "gaussian_mean": gaussian_mean,
        "gaussian_std": gaussian_std,
        "num_drop_columns": num_drop_columns,
        "dropped_column_indices": dropped_column_indices,
        "feature_index_sets_used": {
            "continuous_feature_indices": continuous_feature_indices,
            "gaussian_feature_indices_used": gaussian_feature_indices_used,
        },
        "shapes": {
            "x_before": list(x_shape_before),
            "x_after": list(x_shape_after),
            "feature_drop_mask": list(mask_shape) if mask_shape is not None else None,
        },
        "mask_saved": save_mask,
        "mask_key": FEATURE_DROP_MASK_KEY if save_mask else None,
        "mask_path": str(mask_path) if mask_path is not None else None,
    }


def save_outputs(
    full_matrix: np.ndarray,
    x: np.ndarray,
    output_path: str | Path,
    metadata: dict[str, Any],
    mask: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Save the modified CSV file and its companion metadata JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_matrix = np.concatenate([full_matrix[:, :CAUSAL_PREFIX_COLUMNS], x], axis=1)
    np.savetxt(output_file, output_matrix, delimiter=",", fmt="%.16g")

    if mask is not None:
        mask_path = output_file.with_name(f"{output_file.stem}_{FEATURE_DROP_MASK_KEY}.csv")
        np.savetxt(mask_path, mask.astype(np.uint8), delimiter=",", fmt="%d")

    metadata_path = output_file.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return output_file, metadata_path


def run_noise_pipeline(
    *,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    seed: int,
    gaussian_mean: float = 0.0,
    gaussian_std: float | None = None,
    num_drop_columns: int | None = None,
    continuous_feature_indices: str | Sequence[int] | None = None,
    save_mask: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full IHDP covariate noise pipeline and save outputs."""
    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {MODE_CHOICES}.")
    if mode in {"gaussian", "both"} and gaussian_std is None:
        raise ValueError("--gaussian_std is required when mode is 'gaussian' or 'both'.")
    if mode in {"drop", "both"} and num_drop_columns is None:
        raise ValueError("--num_drop_columns is required when mode is 'drop' or 'both'.")

    full_matrix, x_original = load_ihdp_csv(input_path)
    parsed_continuous_indices = parse_index_list(continuous_feature_indices)
    validate_feature_indices(
        n_features=x_original.shape[1],
        continuous_feature_indices=parsed_continuous_indices,
    )

    rng = np.random.default_rng(seed)
    drop_mask = np.zeros(x_original.shape[1], dtype=bool)
    dropped_column_indices: list[int] = []
    gaussian_feature_indices_used = parsed_continuous_indices

    if verbose:
        print(f"Loaded {input_path} with X shape {x_original.shape}.")

    if mode == "gaussian":
        x_processed = apply_gaussian_noise(
            x=x_original,
            rng=rng,
            gaussian_mean=gaussian_mean,
            gaussian_std=float(gaussian_std),
            continuous_feature_indices=gaussian_feature_indices_used,
        )
    elif mode == "drop":
        x_processed, drop_mask, dropped_column_indices = drop_feature_columns(
            x=x_original,
            rng=rng,
            num_drop_columns=int(num_drop_columns),
        )
    else:
        x_processed, drop_mask, dropped_column_indices = apply_combined_noise(
            x=x_original,
            rng=rng,
            gaussian_mean=gaussian_mean,
            gaussian_std=float(gaussian_std),
            num_drop_columns=int(num_drop_columns),
            continuous_feature_indices=gaussian_feature_indices_used,
        )

    output_mask = drop_mask if save_mask else None
    metadata = build_metadata(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        seed=seed,
        gaussian_mean=gaussian_mean,
        gaussian_std=gaussian_std,
        num_drop_columns=num_drop_columns,
        dropped_column_indices=dropped_column_indices,
        continuous_feature_indices=parsed_continuous_indices,
        gaussian_feature_indices_used=gaussian_feature_indices_used,
        x_shape_before=x_original.shape,
        x_shape_after=x_processed.shape,
        save_mask=save_mask,
        mask_shape=drop_mask.shape if save_mask else None,
    )
    saved_output_path, metadata_path = save_outputs(
        full_matrix=full_matrix,
        x=x_processed,
        output_path=output_path,
        metadata=metadata,
        mask=output_mask,
    )

    if verbose:
        print(f"Saved modified dataset to {saved_output_path}.")
        print(f"Saved metadata to {metadata_path}.")

    return {
        "matrix": full_matrix,
        "x": x_processed,
        "metadata": metadata,
        "output_path": saved_output_path,
        "metadata_path": metadata_path,
        "mask": output_mask,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Inject controlled covariate noise into an IHDP CSV replica "
            "for robust causal learning experiments."
        )
    )
    parser.add_argument("--input_path", required=True, help="Path to the input IHDP CSV file.")
    parser.add_argument("--output_path", required=True, help="Path to save the modified CSV file.")
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES, help="Perturbation mode.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--gaussian_mean",
        type=float,
        default=0.0,
        help="Mean of Gaussian noise. Default: 0.0.",
    )
    parser.add_argument(
        "--gaussian_std",
        type=float,
        default=None,
        help="Standard deviation of Gaussian noise.",
    )
    parser.add_argument(
        "--num_drop_columns",
        type=int,
        default=None,
        help="Number of covariate columns to drop.",
    )
    parser.add_argument(
        "--continuous_feature_indices",
        default=None,
        help="Optional comma-separated feature indices to receive Gaussian noise.",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save the feature-drop mask to a sibling CSV file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information while running.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_noise_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            seed=args.seed,
            gaussian_mean=args.gaussian_mean,
            gaussian_std=args.gaussian_std,
            num_drop_columns=args.num_drop_columns,
            continuous_feature_indices=args.continuous_feature_indices,
            save_mask=args.save_mask,
            verbose=args.verbose,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
