from __future__ import annotations
import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymc as pm
import pymc_bart as pmb

warnings.filterwarnings("ignore", category=UserWarning)


# Data
CEVAE_BASE = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/"
    "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/IHDP"
)

DATA_DIR = Path("../data/ihdp_dataset/csv")

# DATA_DIR = Path("../experiments/knn_counterfactual/noisy/gaussianSTD_test")
# DATA_DIR = Path("../experiments/knn_counterfactual/noisy/drop_3_rep")
# DATA_DIR = Path("../experiments/knn_counterfactual/noisy/both_noise")

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


def load_replica(path: Path) -> IHDPDataset:
    # Detect if first row is a header or raw data
    with open(path) as f:
        first = f.readline().split(",")[0].strip()
    has_header = not _is_float(first)

    if has_header:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        t   = df["treatment"].to_numpy().astype(int)
        yf  = df["y_factual"].to_numpy().astype(float)
        ycf = df["y_cfactual"].to_numpy().astype(float)
        mu0 = df["mu0"].to_numpy().astype(float)
        mu1 = df["mu1"].to_numpy().astype(float)
        meta = {"treatment", "y_factual", "y_cfactual", "mu0", "mu1"}
        x_cols = [c for c in df.columns if c not in meta]
        x = df[x_cols].to_numpy().astype(float)
    else:
        # CEVAE format: no header, columns are t, yf, ycf, mu0, mu1, x1..x25
        data = np.loadtxt(path, delimiter=",", dtype=float)
        t   = data[:, 0].astype(int)
        yf  = data[:, 1]
        ycf = data[:, 2]
        mu0 = data[:, 3]
        mu1 = data[:, 4]
        x   = data[:, 5:]

    return IHDPDataset(path=path, treatment=t, y_factual=yf,
                       y_cfactual=ycf, mu0=mu0, mu1=mu1, x=x)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_all_replicas(data_dir: Path = DATA_DIR) -> list[IHDPDataset]:
    paths = sorted(data_dir.glob("ihdp_npci_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No ihdp_npci_*.csv files found in {data_dir}")
    datasets = [load_replica(p) for p in paths]
    print(f"Loaded {len(datasets)} replicas from {data_dir} — "
          f"{datasets[0].x.shape[0]} units, {datasets[0].x.shape[1]} covariates each")
    return datasets


# Metrics
def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_metrics(
    tau_pred: np.ndarray,
    mu0_true: np.ndarray,
    mu1_true: np.ndarray,
    y0_pred: np.ndarray,
    y1_pred: np.ndarray,
    y_factual: np.ndarray,
    y_cfactual: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, float]:
    tau_true = mu1_true - mu0_true
    treated_mask = treatment == 1
    control_mask = ~treated_mask

    pehe = rmse(tau_pred, tau_true)
    ate_error = float(abs(tau_pred.mean() - tau_true.mean()))

    if treated_mask.sum() == 0:
        att_error = float("nan")
    else:
        att_error = float(abs(
            tau_pred[treated_mask].mean() - tau_true[treated_mask].mean()
        ))

    policy_pred = (tau_pred > 0).astype(int)
    policy_true = (tau_true > 0).astype(int)
    policy_value = float((policy_pred == policy_true).mean())

    # Counterfactual RMSE — matches KNN teammate output
    y_cf_hat = np.where(treated_mask, y0_pred, y1_pred)
    cf_rmse = rmse(y_cf_hat, y_cfactual)
    control_cf_rmse = rmse(y_cf_hat[control_mask], y_cfactual[control_mask])
    treated_cf_rmse = rmse(y_cf_hat[treated_mask], y_cfactual[treated_mask])

    return {
        "pehe":            round(pehe, 4),
        "ate_error":       round(ate_error, 4),
        "att_error":       round(att_error, 4),
        "policy_value":    round(policy_value, 4),
        "cf_rmse":         round(cf_rmse, 4),
        "control_cf_rmse": round(control_cf_rmse, 4),
        "treated_cf_rmse": round(treated_cf_rmse, 4),
    }


# BART — Hill (2011): fit with T as covariate, predict counterfactuals by toggling T
def run_bart_on_replica(
    dataset: IHDPDataset,
    n_trees: int = 50,
    draws: int = 500,
    tune: int = 500,
    random_seed: int = 42,
    alpha: float = 0.95,
    beta: float = 2.0,
) -> dict[str, float]:

    X = dataset.x
    T = dataset.treatment.astype(float)
    Y = dataset.y_factual

    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    idx_tr = idx[:n_tr]
    idx_te = idx[n_tr:]

    X_tr, T_tr, Y_tr = X[idx_tr], T[idx_tr], Y[idx_tr]
    X_te, T_te = X[idx_te], T[idx_te]
    mu0_te, mu1_te = dataset.mu0[idx_te], dataset.mu1[idx_te]
    y_factual_te = dataset.y_factual[idx_te]
    y_cfactual_te = dataset.y_cfactual[idx_te]

    XT_tr = np.column_stack([X_tr, T_tr])
    scaler = StandardScaler()
    XT_tr_s = scaler.fit_transform(XT_tr)

    X_te_t1 = scaler.transform(np.column_stack([X_te, np.ones(len(X_te))]))
    X_te_t0 = scaler.transform(np.column_stack([X_te, np.zeros(len(X_te))]))

    n_te = len(X_te)

    # Combined X: [train | test_T1 | test_T0] — pymc-bart 0.9.x has no predict fn
    X_all = np.vstack([XT_tr_s, X_te_t1, X_te_t0])
    Y_all = np.concatenate([Y_tr, np.zeros(2 * n_te)])

    with pm.Model() as bart_model:
        mu_all = pmb.BART("mu_all", X=X_all, Y=Y_all, m=n_trees, alpha=alpha, beta=beta)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y_obs", mu=mu_all[:n_tr], sigma=sigma, observed=Y_tr)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            cores=1,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )

    mu_post = idata.posterior["mu_all"].values  # type: ignore[union-attr]
    y1_pred = mu_post[:, :, n_tr:n_tr + n_te].mean(axis=(0, 1))
    y0_pred = mu_post[:, :, n_tr + n_te:].mean(axis=(0, 1))
    tau_pred = y1_pred - y0_pred

    return compute_metrics(
        tau_pred=tau_pred,
        mu0_true=mu0_te,
        mu1_true=mu1_te,
        y0_pred=y0_pred,
        y1_pred=y1_pred,
        y_factual=y_factual_te,
        y_cfactual=y_cfactual_te,
        treatment=T_te,
    )


# Run all
def run_all(
    n_replicas: int = 10,
    n_trees: int = 50,
    draws: int = 500,
    tune: int = 500,
    out_csv: str = "bart_results.csv",
) -> pd.DataFrame:

    print("\n" + "="*60)
    print(" BART on IHDP")
    print(f" Replicas    : {n_replicas}")
    print(f" Trees       : {n_trees}")
    print(f" MCMC draws  : {draws}  tune: {tune}")
    print("="*60 + "\n")

    datasets = load_all_replicas()
    datasets = datasets[:n_replicas]
    all_results = []

    for i, dataset in enumerate(datasets):
        print(f"\n── Replica {i+1}/{n_replicas} ({dataset.path.name}) ──")
        try:
            metrics = run_bart_on_replica(
                dataset,
                n_trees=n_trees,
                draws=draws,
                tune=tune,
                random_seed=42 + i,
            )
            row: dict[str, object] = {"replica": dataset.path.name, "status": "ok", **metrics}
            print(f"  PEHE={metrics['pehe']:.4f}  "
                  f"ATE_err={metrics['ate_error']:.4f}  "
                  f"ATT_err={metrics['att_error']:.4f}  "
                  f"Policy={metrics['policy_value']:.4f}  "
                  f"CF_RMSE={metrics['cf_rmse']:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            row = {
                "replica": dataset.path.name, "status": f"error: {e}",
                "pehe": np.nan, "ate_error": np.nan, "att_error": np.nan,
                "policy_value": np.nan, "cf_rmse": np.nan,
                "control_cf_rmse": np.nan, "treated_cf_rmse": np.nan,
            }
        all_results.append(row)

    metric_cols = ["pehe", "ate_error", "att_error", "policy_value",
                   "cf_rmse", "control_cf_rmse", "treated_cf_rmse"]

    df = pd.DataFrame(all_results)
    ok = df[df["status"] == "ok"][metric_cols]

    # Append summary rows to CSV
    mean_row: dict[str, object] = {"replica": "MEAN", "status": ""}
    std_row: dict[str, object] = {"replica": "STD", "status": ""}
    median_row: dict[str, object] = {"replica": "MEDIAN", "status": ""}
    for col in metric_cols:
        mean_row[col] = round(float(ok[col].mean()), 4)
        std_row[col] = round(float(ok[col].std()), 4)
        median_row[col] = round(float(ok[col].median()), 4)

    df_out = pd.concat([df, pd.DataFrame([mean_row, std_row, median_row])], ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    print("\n" + "="*60)
    print(" BART RESULTS — across replicas")
    print("="*60)
    for col in metric_cols:
        m = ok[col].mean()
        s = ok[col].std()
        med = ok[col].median()
        print(f"  {col.upper().replace('_', ' '):<22} mean={m:.4f}  std={s:.4f}  median={med:.4f}")
    print("="*60)
    print("\n  Literature reference (Hill 2011): PEHE ≈ 2.1")

    return df


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART on IHDP")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--beta",  type=float, default=2.0)
    args = parser.parse_args()

    if args.test:
        print("TEST MODE: 1 replica, reduced MCMC")
        run_all(n_replicas=1, draws=200, tune=200, n_trees=20,
                out_csv="bart_results_test.csv")
    else:
        run_all(
            n_replicas=args.n,
            draws=args.draws,
            tune=args.tune,
            n_trees=args.trees,
        )