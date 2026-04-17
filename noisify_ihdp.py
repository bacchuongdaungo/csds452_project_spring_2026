"""
IHDP covariate noise injection for robust causal learning experiments.

Usage examples:
1. Gaussian only
   python noisify_ihdp.py --input_path ihdp.npz --output_path ihdp_gaussian.npz --mode gaussian --gaussian_std 0.1 --seed 42

2. Feature drop only
   python noisify_ihdp.py --input_path ihdp.npz --output_path ihdp_drop.npz --mode drop --num_drop_columns 5 --drop_scope global --seed 42

3. Combined mode
   python noisify_ihdp.py --input_path ihdp.npz --output_path ihdp_both.npz --mode both --gaussian_std 0.1 --num_drop_columns 5 --drop_scope per_replication --continuous_only_for_gaussian --continuous_feature_indices 0,1,2,3,4,5 --seed 42 --save_mask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


FEATURE_DROP_MASK_KEY = "feature_drop_mask"
LAYOUT_CHOICES = ("auto", "nsf", "snr", "rns")
MODE_CHOICES = ("gaussian", "drop", "both")
DROP_SCOPE_CHOICES = ("global", "per_replication")


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


def load_ihdp_data(input_path: str | Path, x_key: str = "x") -> dict[str, np.ndarray]:
    """Load an IHDP-style .npz file and return its arrays keyed by name."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise ValueError(f"Input file does not exist: {input_file}")
    if input_file.suffix.lower() != ".npz":
        raise ValueError(f"Input file must be a .npz archive: {input_file}")

    with np.load(input_file, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    if x_key not in arrays:
        available = ", ".join(sorted(arrays))
        raise ValueError(f"Covariate key '{x_key}' was not found in input. Available keys: {available}")

    x = arrays[x_key]
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"Covariate array '{x_key}' must be numeric. Found dtype {x.dtype}.")
    return arrays


def detect_layout(x: np.ndarray, layout: str = "auto") -> str:
    """Detect or validate the covariate layout."""
    if layout not in LAYOUT_CHOICES:
        raise ValueError(f"Unsupported layout '{layout}'. Choose from {LAYOUT_CHOICES}.")

    if layout != "auto":
        if layout == "nsf" and x.ndim != 2:
            raise ValueError(f"Layout 'nsf' expects a 2D covariate array. Found shape {x.shape}.")
        if layout in {"snr", "rns"} and x.ndim != 3:
            raise ValueError(f"Layout '{layout}' expects a 3D covariate array. Found shape {x.shape}.")
        return layout

    if x.ndim == 2:
        return "nsf"
    if x.ndim != 3:
        raise ValueError(
            f"Auto layout detection supports only 2D or 3D covariates. Found shape {x.shape}."
        )

    shape = x.shape
    smallest_axis = int(np.argmin(shape))
    min_size = shape[smallest_axis]
    is_unique_smallest = sum(int(size == min_size) for size in shape) == 1
    if smallest_axis == 1 and is_unique_smallest:
        return "snr"
    if smallest_axis == 2 and is_unique_smallest:
        return "rns"

    raise ValueError(
        "Could not safely auto-detect 3D covariate layout from shape "
        f"{shape}. Use --layout snr or --layout rns explicitly."
    )


def normalize_x_layout(x: np.ndarray, layout: str) -> np.ndarray:
    """Convert X into the canonical shape (n_replications, n_samples, n_features)."""
    if layout == "nsf":
        return x[np.newaxis, :, :]
    if layout == "snr":
        return np.transpose(x, (2, 0, 1))
    if layout == "rns":
        return np.array(x, copy=True)
    raise ValueError(f"Unsupported layout '{layout}'.")


def restore_x_layout(x: np.ndarray, layout: str) -> np.ndarray:
    """Restore X from canonical (n_replications, n_samples, n_features) shape."""
    if layout == "nsf":
        return x[0]
    if layout == "snr":
        return np.transpose(x, (1, 2, 0))
    if layout == "rns":
        return x
    raise ValueError(f"Unsupported layout '{layout}'.")


def _validate_feature_indices(
    n_features: int,
    binary_feature_indices: list[int] | None,
    continuous_feature_indices: list[int] | None,
) -> None:
    """Validate feature index sets against the covariate dimension."""
    for name, indices in (
        ("binary_feature_indices", binary_feature_indices),
        ("continuous_feature_indices", continuous_feature_indices),
    ):
        if indices is None:
            continue
        out_of_range = [index for index in indices if index >= n_features]
        if out_of_range:
            raise ValueError(
                f"{name} contains out-of-range indices {out_of_range}. "
                f"Valid feature indices are 0 to {n_features - 1}."
            )

    if binary_feature_indices is not None and continuous_feature_indices is not None:
        overlap = sorted(set(binary_feature_indices).intersection(continuous_feature_indices))
        if overlap:
            raise ValueError(
                f"binary_feature_indices and continuous_feature_indices overlap at {overlap}."
            )


def _infer_continuous_feature_indices(
    x_rns: np.ndarray, binary_feature_indices: list[int] | None
) -> list[int]:
    """Infer continuous features conservatively by excluding binary-looking columns."""
    n_features = x_rns.shape[2]
    if binary_feature_indices is not None:
        continuous = [index for index in range(n_features) if index not in set(binary_feature_indices)]
        if not continuous:
            raise ValueError(
                "No continuous features remain after excluding binary_feature_indices."
            )
        return continuous

    continuous: list[int] = []
    for index in range(n_features):
        values = x_rns[:, :, index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        if np.all(np.isin(finite_values, [0, 1])):
            continue
        continuous.append(index)

    if not continuous:
        raise ValueError(
            "Could not safely infer continuous_feature_indices. "
            "Provide --continuous_feature_indices or --binary_feature_indices."
        )
    return continuous


def _restore_output_dtype(
    x_processed: np.ndarray, original_dtype: np.dtype[Any], mode: str
) -> np.ndarray:
    """Preserve the original floating dtype when doing so is safe."""
    if np.issubdtype(original_dtype, np.floating):
        return x_processed.astype(original_dtype, copy=False)
    if mode == "drop" and np.issubdtype(original_dtype, np.integer):
        fractional, _ = np.modf(x_processed)
        if np.allclose(fractional, 0.0):
            return x_processed.astype(original_dtype, copy=False)
    return x_processed


def apply_gaussian_noise(
    x_rns: np.ndarray,
    rng: np.random.Generator,
    gaussian_mean: float,
    gaussian_std: float,
    continuous_feature_indices: list[int] | None = None,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Add Gaussian noise to the selected covariate entries only."""
    if gaussian_std < 0:
        raise ValueError("gaussian_std must be non-negative.")

    x_noisy = np.array(x_rns, dtype=np.float64, copy=True)
    apply_mask = np.ones(x_noisy.shape, dtype=bool)

    if continuous_feature_indices is not None:
        feature_mask = np.zeros(x_noisy.shape[2], dtype=bool)
        feature_mask[continuous_feature_indices] = True
        apply_mask &= feature_mask[np.newaxis, np.newaxis, :]

    if protected_mask is not None:
        if protected_mask.shape != (x_noisy.shape[0], x_noisy.shape[2]):
            raise ValueError(
                "protected_mask must have shape (n_replications, n_features)."
            )
        apply_mask &= ~protected_mask[:, np.newaxis, :]

    noise = rng.normal(loc=gaussian_mean, scale=gaussian_std, size=x_noisy.shape)
    x_noisy[apply_mask] = x_noisy[apply_mask] + noise[apply_mask]
    return x_noisy


def apply_feature_drop(
    x_rns: np.ndarray,
    rng: np.random.Generator,
    num_drop_columns: int,
    drop_value: float,
    drop_scope: str,
) -> tuple[np.ndarray, np.ndarray, list[int] | list[list[int]]]:
    """Drop covariate columns consistently across samples within an experiment."""
    if drop_scope not in DROP_SCOPE_CHOICES:
        raise ValueError(f"drop_scope must be one of {DROP_SCOPE_CHOICES}.")

    n_replications, _, n_features = x_rns.shape
    if num_drop_columns < 0 or num_drop_columns > n_features:
        raise ValueError(
            f"num_drop_columns must be between 0 and {n_features}. "
            f"Received {num_drop_columns}."
        )

    x_dropped = np.array(x_rns, dtype=np.float64, copy=True)
    drop_mask = np.zeros((n_replications, n_features), dtype=bool)

    if num_drop_columns == 0:
        empty = [] if drop_scope == "global" else [[] for _ in range(n_replications)]
        return x_dropped, drop_mask, empty

    if drop_scope == "global":
        # Global drop keeps the same observed covariate space for the full benchmark run.
        dropped_columns = np.sort(
            rng.choice(n_features, size=num_drop_columns, replace=False)
        ).tolist()
        drop_mask[:, dropped_columns] = True
        keep_columns = np.setdiff1d(np.arange(n_features), dropped_columns)
        x_dropped = x_dropped[:, :, keep_columns]
        return x_dropped, drop_mask, dropped_columns

    per_replication_columns: list[list[int]] = []
    for replication_index in range(n_replications):
        # Per-replication drop changes covariate availability across replications,
        # but keeps it fixed within each replication so the experiment remains coherent.
        dropped_columns = np.sort(
            rng.choice(n_features, size=num_drop_columns, replace=False)
        ).tolist()
        drop_mask[replication_index, dropped_columns] = True
        x_dropped[replication_index, :, dropped_columns] = drop_value
        per_replication_columns.append(dropped_columns)

    return x_dropped, drop_mask, per_replication_columns


def apply_combined_noise(
    x_rns: np.ndarray,
    rng: np.random.Generator,
    gaussian_mean: float,
    gaussian_std: float,
    num_drop_columns: int,
    drop_value: float,
    drop_scope: str,
    continuous_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int] | list[list[int]]]:
    """Apply structured feature drop first, then add Gaussian noise to visible covariates."""
    x_dropped, drop_mask, dropped_columns = apply_feature_drop(
        x_rns=x_rns,
        rng=rng,
        num_drop_columns=num_drop_columns,
        drop_value=drop_value,
        drop_scope=drop_scope,
    )
    x_noisy = apply_gaussian_noise(
        x_rns=x_dropped,
        rng=rng,
        gaussian_mean=gaussian_mean,
        gaussian_std=gaussian_std,
        continuous_feature_indices=continuous_feature_indices,
        protected_mask=drop_mask,
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
    dropped_column_indices: list[int] | list[list[int]],
    drop_scope: str | None,
    layout_used: str,
    x_key_used: str,
    continuous_only_for_gaussian: bool,
    binary_feature_indices: list[int] | None,
    continuous_feature_indices: list[int] | None,
    gaussian_feature_indices_used: list[int] | None,
    x_shape_before: Sequence[int],
    x_shape_after: Sequence[int],
    normalized_x_shape: Sequence[int],
    save_mask: bool,
    mask_shape: Sequence[int] | None,
) -> dict[str, Any]:
    """Create a JSON-serializable description of the perturbation run."""
    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "mode": mode,
        "seed": int(seed),
        "gaussian_mean": gaussian_mean,
        "gaussian_std": gaussian_std,
        "num_drop_columns": num_drop_columns,
        "dropped_column_indices": dropped_column_indices,
        "drop_scope": drop_scope,
        "layout_used": layout_used,
        "x_key_used": x_key_used,
        "continuous_only_for_gaussian": continuous_only_for_gaussian,
        "feature_index_sets_used": {
            "binary_feature_indices": binary_feature_indices,
            "continuous_feature_indices": continuous_feature_indices,
            "gaussian_feature_indices_used": gaussian_feature_indices_used,
        },
        "shapes": {
            "x_before": list(x_shape_before),
            "x_after": list(x_shape_after),
            "x_normalized": list(normalized_x_shape),
            "feature_drop_mask": list(mask_shape) if mask_shape is not None else None,
        },
        "mask_saved": save_mask,
        "mask_key": FEATURE_DROP_MASK_KEY if save_mask else None,
    }


def save_outputs(
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
    metadata: dict[str, Any],
    mask: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Save the modified .npz file and its companion metadata JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    arrays_to_save = dict(arrays)
    if mask is not None:
        arrays_to_save[FEATURE_DROP_MASK_KEY] = mask.astype(np.uint8)

    np.savez_compressed(output_file, **arrays_to_save)

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
    drop_value: float = 0.0,
    drop_scope: str = "global",
    x_key: str = "x",
    layout: str = "auto",
    continuous_only_for_gaussian: bool = False,
    binary_feature_indices: str | Sequence[int] | None = None,
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

    arrays = load_ihdp_data(input_path=input_path, x_key=x_key)
    x_original = arrays[x_key]
    layout_used = detect_layout(x_original, layout=layout)
    x_rns = normalize_x_layout(x_original, layout=layout_used)

    n_features = x_rns.shape[2]
    parsed_binary_indices = parse_index_list(binary_feature_indices)
    parsed_continuous_indices = parse_index_list(continuous_feature_indices)
    _validate_feature_indices(
        n_features=n_features,
        binary_feature_indices=parsed_binary_indices,
        continuous_feature_indices=parsed_continuous_indices,
    )

    gaussian_feature_indices_used: list[int] | None = None
    if continuous_only_for_gaussian:
        gaussian_feature_indices_used = (
            parsed_continuous_indices
            if parsed_continuous_indices is not None
            else _infer_continuous_feature_indices(x_rns, parsed_binary_indices)
        )
    elif parsed_continuous_indices is not None:
        gaussian_feature_indices_used = parsed_continuous_indices

    rng = np.random.default_rng(seed)
    drop_mask = np.zeros((x_rns.shape[0], x_rns.shape[2]), dtype=bool)
    dropped_column_indices: list[int] | list[list[int]] = []

    if verbose:
        print(f"Loaded {input_path} with X shape {x_original.shape} and layout '{layout_used}'.")

    if mode == "gaussian":
        x_processed = apply_gaussian_noise(
            x_rns=x_rns,
            rng=rng,
            gaussian_mean=gaussian_mean,
            gaussian_std=float(gaussian_std),
            continuous_feature_indices=gaussian_feature_indices_used,
        )
    elif mode == "drop":
        x_processed, drop_mask, dropped_column_indices = apply_feature_drop(
            x_rns=x_rns,
            rng=rng,
            num_drop_columns=int(num_drop_columns),
            drop_value=drop_value,
            drop_scope=drop_scope,
        )
    else:
        x_processed, drop_mask, dropped_column_indices = apply_combined_noise(
            x_rns=x_rns,
            rng=rng,
            gaussian_mean=gaussian_mean,
            gaussian_std=float(gaussian_std),
            num_drop_columns=int(num_drop_columns),
            drop_value=drop_value,
            drop_scope=drop_scope,
            continuous_feature_indices=gaussian_feature_indices_used,
        )

    x_restored = restore_x_layout(x_processed, layout=layout_used)
    x_restored = _restore_output_dtype(x_restored, x_original.dtype, mode=mode)
    output_arrays = dict(arrays)
    output_arrays[x_key] = x_restored
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
        drop_scope=drop_scope if mode in {"drop", "both"} else None,
        layout_used=layout_used,
        x_key_used=x_key,
        continuous_only_for_gaussian=continuous_only_for_gaussian,
        binary_feature_indices=parsed_binary_indices,
        continuous_feature_indices=parsed_continuous_indices,
        gaussian_feature_indices_used=gaussian_feature_indices_used,
        x_shape_before=x_original.shape,
        x_shape_after=x_restored.shape,
        normalized_x_shape=x_rns.shape,
        save_mask=save_mask,
        mask_shape=drop_mask.shape if save_mask else None,
    )

    saved_output_path, metadata_path = save_outputs(
        arrays=output_arrays,
        output_path=output_path,
        metadata=metadata,
        mask=output_mask,
    )

    if verbose:
        print(f"Saved modified dataset to {saved_output_path}.")
        print(f"Saved metadata to {metadata_path}.")

    return {
        "arrays": output_arrays,
        "metadata": metadata,
        "output_path": saved_output_path,
        "metadata_path": metadata_path,
        "mask": output_mask,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Inject controlled covariate noise into an IHDP-style .npz file "
            "for robust causal learning experiments."
        )
    )
    parser.add_argument("--input_path", required=True, help="Path to the input IHDP .npz file.")
    parser.add_argument("--output_path", required=True, help="Path to save the modified .npz file.")
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
        "--drop_value",
        type=float,
        default=0.0,
        help="Value used for dropped covariate columns. Default: 0.0.",
    )
    parser.add_argument(
        "--drop_scope",
        choices=DROP_SCOPE_CHOICES,
        default="global",
        help="Whether dropped columns are shared globally or vary by replication.",
    )
    parser.add_argument(
        "--x_key",
        default="x",
        help="Covariate array key inside the .npz archive. Default: x.",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUT_CHOICES,
        default="auto",
        help="Covariate layout. Default: auto.",
    )
    parser.add_argument(
        "--continuous_only_for_gaussian",
        action="store_true",
        help="Restrict Gaussian noise to continuous covariates only.",
    )
    parser.add_argument(
        "--binary_feature_indices",
        default=None,
        help="Optional comma-separated binary feature indices.",
    )
    parser.add_argument(
        "--continuous_feature_indices",
        default=None,
        help="Optional comma-separated continuous feature indices.",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save the feature-drop mask into the output .npz.",
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
            drop_value=args.drop_value,
            drop_scope=args.drop_scope,
            x_key=args.x_key,
            layout=args.layout,
            continuous_only_for_gaussian=args.continuous_only_for_gaussian,
            binary_feature_indices=args.binary_feature_indices,
            continuous_feature_indices=args.continuous_feature_indices,
            save_mask=args.save_mask,
            verbose=args.verbose,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
