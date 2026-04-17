"""
bart_tune.py
============
Run once to find optimal BART hyperparameters using Optuna.
Tunes on replica 1 only, then you apply the best params to all runs.

Usage:
    python3 bart_tune.py           # 30 trials (recommended)
    python3 bart_tune.py --trials 50
    python3 bart_tune.py --data ../data/ihdp_dataset/csv

Output:
    bart_best_params.json  — best params to copy into your Makefile
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path("../data/ihdp_dataset/csv")


# Inline loader — same logic as bart_ihdp.py, no import dependency
def load_first_replica(data_dir: Path) -> tuple:
    import pandas as pd
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    path = paths[0]
    print(f"Tuning on: {path.name}")

    with open(path) as f:
        first = f.readline().split(",")[0].strip()

    try:
        float(first)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        t   = df["treatment"].to_numpy().astype(int)
        yf  = df["y_factual"].to_numpy().astype(float)
        mu0 = df["mu0"].to_numpy().astype(float)
        mu1 = df["mu1"].to_numpy().astype(float)
        meta = {"treatment", "y_factual", "y_cfactual", "mu0", "mu1"}
        x = df[[c for c in df.columns if c not in meta]].to_numpy().astype(float)
    else:
        data = np.loadtxt(path, delimiter=",", dtype=float)
        t   = data[:, 0].astype(int)
        yf  = data[:, 1]
        mu0 = data[:, 3]
        mu1 = data[:, 4]
        x   = data[:, 5:]

    return x, t, yf, mu0, mu1


def evaluate(x, t, yf, mu0, mu1, n_trees, alpha, beta,
             draws=300, tune=300, seed=42) -> float:
    import pymc as pm
    import pymc_bart as pmb

    T = t.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_tr = int(0.8 * len(x))
    idx_tr, idx_te = idx[:n_tr], idx[n_tr:]

    X_tr, T_tr, Y_tr = x[idx_tr], T[idx_tr], yf[idx_tr]
    X_te = x[idx_te]
    mu0_te, mu1_te = mu0[idx_te], mu1[idx_te]

    XT_tr = np.column_stack([X_tr, T_tr])
    scaler = StandardScaler()
    XT_tr_s = scaler.fit_transform(XT_tr)
    X_te_t1 = scaler.transform(np.column_stack([X_te, np.ones(len(X_te))]))
    X_te_t0 = scaler.transform(np.column_stack([X_te, np.zeros(len(X_te))]))

    n_te = len(X_te)
    X_all = np.vstack([XT_tr_s, X_te_t1, X_te_t0])
    Y_all = np.concatenate([Y_tr, np.zeros(2 * n_te)])

    with pm.Model() as model:
        mu_all = pmb.BART("mu_all", X=X_all, Y=Y_all, m=n_trees,
                          alpha=alpha, beta=beta)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y_obs", mu=mu_all[:n_tr], sigma=sigma, observed=Y_tr)
        idata = pm.sample(draws=draws, tune=tune, cores=1,
                          random_seed=seed, progressbar=False,
                          return_inferencedata=True)

    mu_post = idata.posterior["mu_all"].values  # type: ignore[union-attr]
    y1_pred = mu_post[:, :, n_tr:n_tr + n_te].mean(axis=(0, 1))
    y0_pred = mu_post[:, :, n_tr + n_te:].mean(axis=(0, 1))
    tau_pred = y1_pred - y0_pred
    tau_true = mu1_te - mu0_te

    pehe = float(np.sqrt(np.mean((tau_pred - tau_true) ** 2)))
    return pehe


def tune(data_dir: Path, n_trials: int = 30) -> dict:
    try:
        import optuna # type: ignore
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Run: pip install optuna")

    x, t, yf, mu0, mu1 = load_first_replica(data_dir)

    def objective(trial):
        n_trees = trial.suggest_categorical("n_trees", [50, 100, 200, 300])
        alpha   = trial.suggest_float("alpha", 0.5, 0.99)
        beta    = trial.suggest_float("beta", 1.0, 3.0)

        pehe = evaluate(x, t, yf, mu0, mu1,
                        n_trees=n_trees, alpha=alpha, beta=beta)
        print(f"  Trial {trial.number+1}/{n_trials} — "
              f"trees={n_trees} alpha={alpha:.3f} beta={beta:.3f} "
              f"→ PEHE={pehe:.4f}")
        return pehe

    print(f"\nRunning {n_trials} Optuna trials (minimizing PEHE)...\n")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best_pehe = study.best_value

    print(f"\n{'='*60}")
    print(" BEST PARAMETERS")
    print(f"{'='*60}")
    print(f"  n_trees : {best['n_trees']}")
    print(f"  alpha   : {best['alpha']:.4f}")
    print(f"  beta    : {best['beta']:.4f}")
    print(f"  PEHE    : {best_pehe:.4f}")
    print(f"{'='*60}")
    print("\nAdd to your Makefile run target:")
    print(f"  --trees {best['n_trees']} --alpha {best['alpha']:.4f} --beta {best['beta']:.4f}")

    out = {
        "n_trees":   best["n_trees"],
        "alpha":     round(best["alpha"], 4),
        "beta":      round(best["beta"], 4),
        "pehe":      round(best_pehe, 4),
        "tuned_on":  str(sorted(data_dir.glob("*.csv"))[0]),
        "n_trials":  n_trials,
    }

    with open("bart_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to: bart_best_params.json")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune BART hyperparameters")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--data", type=str, default=str(DATA_DIR),
                        help="Path to data directory")
    args = parser.parse_args()

    tune(Path(args.data), n_trials=args.trials)