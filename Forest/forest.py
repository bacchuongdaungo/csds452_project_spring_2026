from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML # type: ignore
warnings.filterwarnings("ignore", category=UserWarning)


FOREST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FOREST_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ihdp_dataset" / "csv"
KNN_COUNTERFACTUAL_PATH = PROJECT_ROOT / "KNN" / "knn_counterfactual.py"


@dataclass(frozen=True)
class IHDPDataset:
    path: Path
    treatment: np.ndarray
    y_factual: np.ndarray
    y_cfactual: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    x: np.ndarray


def load_replica(path: Path) -> IHDPDataset:
    with open(path) as f:
        first_cell = f.readline().strip().split(",")[0]
    try:
        float(first_cell)
        skip = 0
    except ValueError:
        skip = 1

    data = np.loadtxt(path, delimiter=",", dtype=float, skiprows=skip)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"Unexpected shape {data.shape} in {path}")

    return IHDPDataset(
        path=path,
        treatment=data[:, 0].astype(int),
        y_factual=data[:, 1],
        y_cfactual=data[:, 2],
        mu0=data[:, 3],
        mu1=data[:, 4],
        x=data[:, 5:],
    )


def load_all_replicas(data_dir: Path) -> list[IHDPDataset]:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No IHDP replicas found in {data_dir}. Expected files like ihdp_npci_1.csv."
        )

    datasets = [load_replica(path) for path in paths]
    print(
        f"Loaded {len(datasets)} IHDP replicas from data/ihdp_dataset "
        f"({datasets[0].x.shape[0]} rows, {datasets[0].x.shape[1]} covariates each)."
    )
    return datasets


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_metrics(
    tau_pred: np.ndarray,
    mu0_true: np.ndarray,
    mu1_true: np.ndarray,
    y0_pred: np.ndarray,
    y1_pred: np.ndarray,
    y_cfactual: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, float]:
    tau_true = mu1_true - mu0_true
    treated_mask = treatment == 1
    control_mask = ~treated_mask

    pehe = rmse(tau_pred, tau_true)
    ate_error = float(abs(tau_pred.mean() - tau_true.mean()))
    att_error = float(abs(tau_pred[treated_mask].mean() - tau_true[treated_mask].mean()))
    policy_value = float(((tau_pred > 0).astype(int) == (tau_true > 0).astype(int)).mean())
    y_cf_hat = np.where(treated_mask, y0_pred, y1_pred)
    cf_rmse = rmse(y_cf_hat, y_cfactual)
    control_cf_rmse = rmse(y_cf_hat[control_mask], y_cfactual[control_mask])
    treated_cf_rmse = rmse(y_cf_hat[treated_mask], y_cfactual[treated_mask])

    return {
        "pehe": round(pehe, 4),
        "ate_error": round(ate_error, 4),
        "att_error": round(att_error, 4),
        "policy_value": round(policy_value, 4),
        "cf_rmse": round(cf_rmse, 4),
        "control_cf_rmse": round(control_cf_rmse, 4),
        "treated_cf_rmse": round(treated_cf_rmse, 4),
    }


def build_base_model(random_state: int, n_estimators: int, min_samples_leaf: int, max_depth: int):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
    )


def configure_local_temp_dir() -> Path:
    temp_dir = FOREST_DIR / "tmp"
    temp_dir.mkdir(exist_ok=True)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    return temp_dir


def load_knn_counterfactual_module():
    module_name = "project_knn_counterfactual"
    spec = importlib.util.spec_from_file_location(module_name, KNN_COUNTERFACTUAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {KNN_COUNTERFACTUAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def estimate_knn_counterfactual_outcomes(
    dataset: IHDPDataset,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    knn_module = load_knn_counterfactual_module()
    knn_dataset = knn_module.load_ihdp_replica(dataset.path)
    knn_result = knn_module.estimate_counterfactuals(knn_dataset)
    return knn_result.y0_hat[test_indices], knn_result.y1_hat[test_indices]


def fit_causal_model(
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
):
  

    model_y = build_base_model(random_state, 100, 3, max_depth=5) # type: ignore
    model_t = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        max_depth=5,
        random_state=random_state,
        n_jobs=1,
    )

    model = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        honest=True,
        inference=False,
        n_jobs=1,
    )
    model.fit(y_train, t_train, X=x_train)
    return model


def estimate_effects(
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
) -> tuple[np.ndarray, str]:
    model = fit_causal_model(
        x_train=x_train,
        t_train=t_train,
        y_train=y_train,
        random_state=random_state,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    tau_pred = model.effect(x_test)
    return tau_pred, "econml_causal_forest"


def run_forest_on_replica(
    dataset: IHDPDataset,
    random_state: int,
    test_size: float,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
) -> tuple[dict[str, float], str]:
    indices = np.arange(len(dataset.x))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.treatment,
    )

    x_train = dataset.x[idx_train]
    t_train = dataset.treatment[idx_train]
    y_train = dataset.y_factual[idx_train]
    x_test = dataset.x[idx_test]
    t_test = dataset.treatment[idx_test]
    y0_pred, y1_pred = estimate_knn_counterfactual_outcomes(dataset, idx_test)

    tau_pred, backend = estimate_effects(
        x_train=x_train,
        t_train=t_train,
        y_train=y_train,
        x_test=x_test,
        random_state=random_state,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )

    metrics = compute_metrics(
        tau_pred=tau_pred,
        mu0_true=dataset.mu0[idx_test],
        mu1_true=dataset.mu1[idx_test],
        y0_pred=y0_pred,
        y1_pred=y1_pred,
        y_cfactual=dataset.y_cfactual[idx_test],
        treatment=t_test,
    )
    return metrics, backend

def run_all(
    data_dir: Path,
    n_replicas: int,
    test_size: float,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
    out_csv: Path,
) -> pd.DataFrame:

    datasets = load_all_replicas(data_dir)[:n_replicas]
    all_results: list[dict[str, object]] = []

    for replica_index, dataset in enumerate(datasets, start=1):
        print(f"Replica {replica_index}/{n_replicas}: {dataset.path.name}")
        metrics, backend = run_forest_on_replica(
            dataset=dataset,
            random_state=42 + replica_index,
            test_size=test_size,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        row: dict[str, object] = {
            "replica": dataset.path.name,
            "status": "ok",
            "backend": backend,
            **metrics,
        }

        all_results.append(row)

    metric_cols = [
        "pehe",
        "ate_error",
        "att_error",
        "policy_value",
        "cf_rmse",
        "control_cf_rmse",
        "treated_cf_rmse",
    ]
    df = pd.DataFrame(all_results)
    ok = df[df["status"] == "ok"][metric_cols]
    summary_rows: list[dict[str, object]] = []
    for label, reducer in (
        ("MEAN", pd.DataFrame.mean),
        ("STD", pd.DataFrame.std),
        ("MEDIAN", pd.DataFrame.median),
    ):
        row: dict[str, object] = {"replica": label, "status": "", "backend": ""}
        for col in metric_cols:
            value = reducer(ok)[col]
            row[col] = round(float(value), 4) if pd.notna(value) else np.nan
        summary_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    df_out.to_csv(out_csv, index=False)


    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("No successful runs.")
    else:
        for col in metric_cols:
            print(
                f"  {col.upper():<18} "
                f"mean={ok[col].mean():.4f} "
                f"std={ok[col].std():.4f} "
                f"median={ok[col].median():.4f}"
            )

    print("\nResults saved to: Forest/forest_results.csv")
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a causal forest style IHDP benchmark from the Forest folder."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n", type=int, default=10, help="Number of IHDP replicas to evaluate.")
    parser.add_argument("--trees", type=int, default=500, help="Number of forest trees.")
    parser.add_argument("--min-leaf", type=int, default=5, help="Minimum samples per leaf.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction.")
    parser.add_argument(
        "--out",
        type=Path,
        default=FOREST_DIR / "forest_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a faster single-replica smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    configure_local_temp_dir()
    args = parse_args()
    noisy_root = PROJECT_ROOT / "experiments" / "knn_counterfactual" / "noisy"
    print("Forest Runs")
    print("Running normal (non-noisy)")
    run_all( #normmal
        data_dir=args.data_dir,
        n_replicas=args.n,
        test_size=args.test_size,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_depth=args.max_depth,
        out_csv=args.out,
    )
    '''
    print("Repeat drop")
    run_all( # Noisy drop
        data_dir= noisy_root / "drop_3_rep",
        n_replicas=args.n,
        test_size=args.test_size,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_depth=args.max_depth,
        out_csv= FOREST_DIR / "forest_drop_repeat.csv",
    )
    '''
    print("gaussian repeat")
    run_all( # Noisy guassian
        data_dir= noisy_root / "gaussianSTD_test",
        n_replicas=args.n,
        test_size=args.test_size,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_depth=args.max_depth,
        out_csv= FOREST_DIR / "forest_gaussianSTD.csv",)
    print("both")
    '''
    run_all( # Noisy both
        data_dir= noisy_root / "both_noise",
        n_replicas=args.n,
        test_size=args.test_size,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_depth=args.max_depth,
        out_csv= FOREST_DIR / "forest_both.csv",
        )
    '''
    


if __name__ == "__main__":
    main()
