"""
forest_tune.py
==============
Run once to find optimal Causal Forest hyperparameters using Optuna.
Tunes on replica 1, apply best params to all runs.

Just press the IDE run button — no CLI args needed.
Results saved to: forest_best_params.json
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")

# ── Config — adjust paths if needed ──────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parent.parent / "data" / "ihdp_dataset" / "csv"
N_TRIALS  = 30   # increase for more thorough search, decrease to go faster
OUT_JSON  = Path(__file__).resolve().parent / "forest_best_params.json"
TEST_SIZE = 0.2


# ── Inline data loader ────────────────────────────────────────────────────────
def load_first_replica(data_dir: Path):
    paths = sorted(data_dir.glob("ihdp_npci_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No ihdp_npci_*.csv files in {data_dir}")
    path = paths[0]
    print(f"Tuning on: {path.name}")
    data = np.loadtxt(path, delimiter=",", dtype=float)
    return (
        data[:, 0].astype(int),   # treatment
        data[:, 1],                # y_factual
        data[:, 3],                # mu0
        data[:, 4],                # mu1
        data[:, 5:],               # x covariates
    )


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_evaluate(t, yf, mu0, mu1, x,
                 n_estimators, min_samples_leaf, max_depth,
                 seed=42) -> float:
    indices = np.arange(len(x))
    idx_tr, idx_te = train_test_split(
        indices, test_size=TEST_SIZE, random_state=seed, stratify=t
    )

    x_tr, t_tr, y_tr = x[idx_tr], t[idx_tr], yf[idx_tr]
    x_te              = x[idx_te]
    mu0_te, mu1_te    = mu0[idx_te], mu1[idx_te]

    model_y = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5,
        random_state=seed, n_jobs=1
    )
    model_t = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        random_state=seed, n_jobs=1
    )
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=seed,
        honest=True,
        inference=False,
        n_jobs=1,
    )
    cf.fit(y_tr, t_tr, X=x_tr)
    tau_pred = cf.effect(x_te)
    tau_true = mu1_te - mu0_te
    pehe = float(math.sqrt(np.mean((tau_pred - tau_true) ** 2)))
    return pehe


# ── Tuning ────────────────────────────────────────────────────────────────────
def tune():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna first: pip install optuna")

    t, yf, mu0, mu1, x = load_first_replica(DATA_DIR)

    def objective(trial):
        n_estimators     = trial.suggest_categorical("n_estimators", [100, 200, 500, 1000])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_depth        = trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20])

        pehe = run_evaluate(t, yf, mu0, mu1, x,
                            n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            max_depth=max_depth)
        print(f"  Trial {trial.number+1}/{N_TRIALS} — "
              f"trees={n_estimators} min_leaf={min_samples_leaf} "
              f"max_depth={max_depth} → PEHE={pehe:.4f}")
        return pehe

    print(f"\nRunning {N_TRIALS} Optuna trials for Causal Forest (minimizing PEHE)...\n")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    best      = study.best_params
    best_pehe = study.best_value

    print(f"\n{'='*50}")
    print(" CAUSAL FOREST BEST PARAMETERS")
    print(f"{'='*50}")
    print(f"  n_estimators     : {best['n_estimators']}")
    print(f"  min_samples_leaf : {best['min_samples_leaf']}")
    print(f"  max_depth        : {best['max_depth']}")
    print(f"  PEHE             : {best_pehe:.4f}")
    print(f"{'='*50}")

    out = {
        "n_estimators":     best["n_estimators"],
        "min_samples_leaf": best["min_samples_leaf"],
        "max_depth":        best["max_depth"],
        "pehe":             round(best_pehe, 4),
        "tuned_on":         str(sorted(DATA_DIR.glob("ihdp_npci_*.csv"))[0]),
        "n_trials":         N_TRIALS,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {OUT_JSON}")
    return out


if __name__ == "__main__":
    tune()