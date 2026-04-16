## IHDP Noise Injection for Robust Causal Learning

### Overview
`noisify_ihdp.py` injects controlled perturbations into IHDP pretreatment covariates `X` and saves the result as a new `.npz` archive plus a companion metadata `.json`.

This is useful when benchmarking robust causal learners under degraded covariate quality, missing confounders, or deployment-time covariate loss. Perturbations are intentionally limited to pretreatment covariates because the goal is to stress the learner's access to confounding information without changing treatment assignment or the underlying outcome-generating process.

### Supported perturbations
- Gaussian noise: adds Gaussian noise to covariates only.
- Random feature drop: drops whole covariate columns, not individual cells.
- Combined mode: performs feature drop first, then adds Gaussian noise to the remaining visible covariates.

### Causal design rationale
Treatment `t` and outcomes `yf`, `ycf`, `mu0`, and `mu1` are never modified. Changing those arrays would alter the benchmark target rather than the observed information available to the model.

Feature drop is column-wise because causal robustness experiments often simulate unavailable covariates, missing confounders, or systematically missing features at deployment. Row-wise random masking is avoided because it changes the observation process in a different way and breaks the stable covariate-space assumption that many causal experiments rely on.

`drop_scope=global` applies the same dropped columns to every replication, which is useful when comparing methods under one shared degradation setting. `drop_scope=per_replication` lets each replication have its own dropped-column subset while still keeping feature availability consistent within that replication.

### Expected input format
The script expects an IHDP-style `.npz` file with a covariate array under `x` by default. It may also contain some subset of:

- `x`
- `t`
- `yf`
- `ycf`
- `mu0`
- `mu1`

Some IHDP files omit some keys. The script is defensive and preserves all original keys, only replacing the chosen covariate key.

Supported covariate layouts:

- `nsf`: `(n_samples, n_features)`
- `snr`: `(n_samples, n_features, n_replications)`
- `rns`: `(n_replications, n_samples, n_features)`
- `auto`: detect layout conservatively when possible

### CLI arguments
- `--input_path`: path to the input IHDP `.npz`
- `--output_path`: path to the modified `.npz`
- `--mode`: one of `gaussian`, `drop`, `both`
- `--seed`: random seed for reproducibility
- `--gaussian_mean`: Gaussian mean, default `0.0`
- `--gaussian_std`: Gaussian standard deviation
- `--num_drop_columns`: number of covariate columns to drop
- `--drop_value`: replacement value for dropped columns, default `0.0`
- `--drop_scope`: `global` or `per_replication`
- `--x_key`: covariate key inside the `.npz`, default `x`
- `--layout`: `auto`, `nsf`, `snr`, or `rns`
- `--continuous_only_for_gaussian`: if set, Gaussian noise is restricted to continuous covariates
- `--binary_feature_indices`: optional comma-separated binary feature indices
- `--continuous_feature_indices`: optional comma-separated continuous feature indices
- `--save_mask`: if set, saves `feature_drop_mask` in the output `.npz`
- `--verbose`: print progress information

### Examples
Gaussian only:

```bash
python noisify_ihdp.py \
  --input_path ihdp.npz \
  --output_path ihdp_gaussian.npz \
  --mode gaussian \
  --gaussian_std 0.1 \
  --seed 42
```

Feature drop only:

```bash
python noisify_ihdp.py \
  --input_path ihdp.npz \
  --output_path ihdp_drop.npz \
  --mode drop \
  --num_drop_columns 5 \
  --drop_scope global \
  --seed 42
```

Combined mode:

```bash
python noisify_ihdp.py \
  --input_path ihdp.npz \
  --output_path ihdp_both.npz \
  --mode both \
  --gaussian_std 0.1 \
  --num_drop_columns 5 \
  --drop_scope per_replication \
  --continuous_only_for_gaussian \
  --continuous_feature_indices 0,1,2,3,4,5 \
  --seed 42 \
  --save_mask
```

### Output files
The tool creates:

- a modified `.npz`
- a companion metadata `.json`
- an optional `feature_drop_mask` array in the output `.npz` when `--save_mask` is used

### Reproducibility
`--seed` controls every random decision in the pipeline, including dropped column selection and Gaussian draws. Reusing the same input, configuration, and seed gives the same result.

### Notes and limitations
- The current implementation uses only NumPy and the Python standard library.
- The perturbations target covariates only and do not simulate missingness mechanisms beyond structured feature drop.
- Natural future extensions include MAR or MNAR missingness, hidden confounding, or noisy outcomes.

### Quick sanity checks
You can verify the generated file with a short Python check:

```python
import json
import numpy as np

original = np.load("ihdp.npz")
modified = np.load("ihdp_both.npz")
metadata = json.load(open("ihdp_both.json", "r", encoding="utf-8"))

assert np.array_equal(original["t"], modified["t"])
assert np.array_equal(original["yf"], modified["yf"])
assert np.array_equal(original["ycf"], modified["ycf"])
assert np.array_equal(original["mu0"], modified["mu0"])
assert np.array_equal(original["mu1"], modified["mu1"])

assert not np.array_equal(original["x"], modified["x"])

drop_mask = modified["feature_drop_mask"]
dropped_columns = metadata["dropped_column_indices"]

# Inspect dropped_columns and confirm dropped columns
# are set to the configured drop_value across the relevant replication scope.
# In combined mode, dropped entries should remain exactly at drop_value,
# which confirms Gaussian noise was not added after dropping.
```
