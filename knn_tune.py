"""
knn_tune.py
===========
Run once to find optimal KNN hyperparameters using Optuna.
Tunes on replica 1, apply best params to all runs.

Place this file next to knn_counterfactual.py and press the IDE run button.
Results saved to: knn_best_params.json
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE.parent / "data" / "ihdp_dataset" / "csv"
N_TRIALS  = 30
OUT_JSON  = HERE / "knn_best_params.json"

# Add project root to path so knn_counterfactual imports work
sys.path.insert(0, str(HERE.parent))

from knn_counterfactual import (  # type: ignore
    load_ihdp_replica,
    estimate_counterfactuals,
)


def tune():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna first: pip install optuna")

    paths = sorted(DATA_DIR.glob("ihdp_npci_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No ihdp_npci_*.csv files in {DATA_DIR}")

    path = paths[0]
    print(f"Tuning on: {path.name}\n")
    dataset = load_ihdp_replica(path)

    def objective(trial):
        k        = trial.suggest_int("k", 1, 20)
        metric   = trial.suggest_categorical("metric", ["euclidean", "manhattan"])
        weighted = trial.suggest_categorical("weighted", [True, False])
        scale    = trial.suggest_categorical("scale", [True, False])

        result = estimate_counterfactuals(
            dataset=dataset,
            k=k,
            metric=metric,
            scale=scale,
            weighted=weighted,
        )
        print(f"  Trial {trial.number+1}/{N_TRIALS} — "
              f"k={k} metric={metric} weighted={weighted} scale={scale} "
              f"→ PEHE={result.pehe:.4f}")
        return result.pehe

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    best      = study.best_params
    best_pehe = study.best_value

    print(f"\n{'='*50}")
    print(" KNN BEST PARAMETERS")
    print(f"{'='*50}")
    print(f"  k        : {best['k']}")
    print(f"  metric   : {best['metric']}")
    print(f"  weighted : {best['weighted']}")
    print(f"  scale    : {best['scale']}")
    print(f"  PEHE     : {best_pehe:.4f}")
    print(f"{'='*50}")
    print(f"\nUse these in your run:")
    print(f"  --k {best['k']} --metric {best['metric']}"
          f"{' --weighted' if best['weighted'] else ''}"
          f"{' --no-scale' if not best['scale'] else ''}")

    out = {
        "k":        best["k"],
        "metric":   best["metric"],
        "weighted": best["weighted"],
        "scale":    best["scale"],
        "pehe":     round(best_pehe, 4),
        "tuned_on": str(path),
        "n_trials": N_TRIALS,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {OUT_JSON}")
    return out


if __name__ == "__main__":
    tune()