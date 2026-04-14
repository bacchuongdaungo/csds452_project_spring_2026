import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


# Load IHDP data from Kaggle
def load_ihdp():
    import kagglehub

    print("Downloading IHDP data from Kaggle...")
    path = kagglehub.dataset_download("konradb/ihdp-data")

    # Find the CSV
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")

    df = pd.read_csv(os.path.join(path, csv_files[0]))
    print(f"Loaded: {df.shape[0]} units, {df.shape[1]} columns")

    # Covariates
    cov_cols = [f"x{i}" for i in range(1, 26)]
    X   = df[cov_cols].values.astype(float)
    T   = df["treatment"].values.astype(float)
    Y   = df["y_factual"].values.astype(float)
    mu0 = df["mu0"].values.astype(float)
    mu1 = df["mu1"].values.astype(float)

    # 80/20 train/test split — fixed seed for reproducibility
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)

    return {
        "X_tr":   X[idx_tr],
        "T_tr":   T[idx_tr],
        "Y_tr":   Y[idx_tr],
        "X_te":   X[idx_te],
        "T_te":   T[idx_te],
        "mu0_te": mu0[idx_te],
        "mu1_te": mu1[idx_te],
    }


# metrics
def compute_metrics(tau_pred, mu0_true, mu1_true, T_test):
    tau_true = mu1_true - mu0_true

    # PEHE
    pehe = np.sqrt(np.mean((tau_pred - tau_true) ** 2))

    # ATE error
    ate_error = np.abs(tau_pred.mean() - tau_true.mean())

    # ATT error
    att_mask = T_test == 1
    if att_mask.sum() == 0:
        att_error = np.nan
    else:
        att_error = np.abs(
            tau_pred[att_mask].mean() - tau_true[att_mask].mean()
        )

    # Policy value
    policy_pred  = (tau_pred > 0).astype(int)
    policy_true  = (tau_true > 0).astype(int)
    policy_value = (policy_pred == policy_true).mean()

    return {
        "pehe":         round(float(pehe), 4),
        "ate_error":    round(float(ate_error), 4),
        "att_error":    round(float(att_error), 4),
        "policy_value": round(float(policy_value), 4),
    }


# Bart Implementation
# Hill (2011): fit BART with T as a covariate, predict counterfactuals by toggling T=1 and T=0 for all test units.
def run_bart(data, n_trees=50, draws=500, tune=500, random_seed=42):
    import pymc as pm
    import pymc_bart as pmb

    X_tr   = data["X_tr"]
    T_tr   = data["T_tr"]
    Y_tr   = data["Y_tr"]
    X_te   = data["X_te"]
    T_te   = data["T_te"]
    mu0_te = data["mu0_te"]
    mu1_te = data["mu1_te"]

    # Stack T as last covariate — Hill (2011)
    XT_tr = np.column_stack([X_tr, T_tr])
    scaler = StandardScaler()
    XT_tr_scaled = scaler.fit_transform(XT_tr)

    # Counterfactual test matrices
    X_te_t1 = scaler.transform(np.column_stack([X_te, np.ones(len(X_te))]))
    X_te_t0 = scaler.transform(np.column_stack([X_te, np.zeros(len(X_te))]))

    n_tr = len(X_tr)
    n_te = len(X_te)

    # Combined X: [train | test_T1 | test_T0]
    X_all = np.vstack([XT_tr_scaled, X_te_t1, X_te_t0])
    Y_all = np.concatenate([Y_tr, np.zeros(2 * n_te)])

    with pm.Model() as bart_model:
        mu_all = pmb.BART("mu_all", X=X_all, Y=Y_all, m=n_trees)
        sigma  = pm.HalfNormal("sigma", sigma=1.0)
        # Likelihood only on training rows
        pm.Normal("y_obs", mu=mu_all[:n_tr], sigma=sigma, observed=Y_tr)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            cores=1,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )

    # Slice test predictions from posterior
    # mu_all shape: (chains, draws, n_tr + 2*n_te)
    mu_posterior = idata.posterior["mu_all"].values  # type: ignore[union-attr]
    mu1_pred = mu_posterior[:, :, n_tr : n_tr + n_te].mean(axis=(0, 1))
    mu0_pred = mu_posterior[:, :, n_tr + n_te :].mean(axis=(0, 1))
    tau_pred = mu1_pred - mu0_pred

    return compute_metrics(tau_pred, mu0_te, mu1_te, T_te)


# run all
def run_all(draws=500, tune=500, n_trees=50, out_csv="bart_results.csv"):

    print("\n" + "="*60)
    print(" BART on IHDP")
    print(f" Trees      : {n_trees}")
    print(f" MCMC draws : {draws}  tune: {tune}")
    print("="*60 + "\n")

    data    = load_ihdp()
    metrics = run_bart(data, n_trees=n_trees, draws=draws, tune=tune)

    print("\n" + "="*60)
    print(" BART RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k.upper().replace('_', ' '):<20} {v:.4f}")
    print("="*60)
    print("\n  Literature reference (Hill 2011): PEHE ≈ 2.1")

    df = pd.DataFrame([metrics])
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    return metrics


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART on IHDP")
    parser.add_argument("--test",  action="store_true")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune",  type=int, default=1000)
    args = parser.parse_args()

    if args.test:
        print("TEST MODE: reduced MCMC")
        run_all(draws=200, tune=200, n_trees=20, out_csv="bart_results_test.csv")
    else:
        run_all(draws=args.draws, tune=args.tune, n_trees=args.trees)