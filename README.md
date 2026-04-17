## IHDP Noise Injection for Robust Causal Learning

### Overview
`noisify_ihdp.py` injects controlled perturbations into IHDP pretreatment covariates `X` in a CSV replica and saves the result as a new `.csv` plus a companion metadata `.json`.

This is useful when benchmarking robust causal learners under degraded covariate quality, missing confounders, or deployment-time covariate loss. Perturbations are intentionally limited to pretreatment covariates because the goal is to stress the learner's access to confounding information without changing treatment assignment or the underlying outcome-generating process.

### Supported perturbations
- Gaussian noise: adds Gaussian noise to covariates only.
- Random feature drop: removes whole covariate columns from the output CSV.
- Combined mode: removes selected covariate columns first, then adds Gaussian noise to the remaining visible covariates.

### Causal design rationale
Treatment `t` and outcomes `yf`, `ycf`, `mu0`, and `mu1` are never modified. Changing those arrays would alter the benchmark target rather than the observed information available to the model.

Feature drop is column-wise because causal robustness experiments often simulate unavailable covariates, missing confounders, or systematically missing features at deployment. Row-wise random masking is avoided because it changes the observation process in a different way and breaks the stable covariate-space assumption that many causal experiments rely on.

Feature drop is intentionally strict: the script samples one subset of covariate columns for the input CSV and removes those columns from the output entirely.

### Expected input format
The script expects one IHDP replica CSV with:

- column `0`: `treatment`
- columns `1` to `4`: `y_factual`, `y_cfactual`, `mu0`, `mu1`
- columns `5+`: pretreatment covariates `X`

Only the covariate columns are modified. The first five causal columns are copied through unchanged.

### CLI arguments
- `--input_path`: path to the input IHDP `.csv`
- `--output_path`: path to the modified `.csv`
- `--mode`: one of `gaussian`, `drop`, `both`
- `--seed`: random seed for reproducibility
- `--gaussian_mean`: Gaussian mean, default `0.0`
- `--gaussian_std`: Gaussian standard deviation
- `--num_drop_columns`: number of covariate columns to drop
- `--continuous_feature_indices`: optional comma-separated feature indices to receive Gaussian noise
- `--save_mask`: if set, saves `feature_drop_mask` to a sibling mask `.csv`
- `--verbose`: print progress information

### Examples
Gaussian only:

```bash
python noisify_ihdp.py \
  --input_path ihdp_npci_1.csv \
  --output_path ihdp_gaussian.csv \
  --mode gaussian \
  --gaussian_std 0.1 \
  --seed 42
```

Feature drop only:

```bash
python noisify_ihdp.py \
  --input_path ihdp_npci_1.csv \
  --output_path ihdp_drop.csv \
  --mode drop \
  --num_drop_columns 5 \
  --seed 42
```

Combined mode:

```bash
python noisify_ihdp.py \
  --input_path ihdp_npci_1.csv \
  --output_path ihdp_both.csv \
  --mode both \
  --gaussian_std 0.1 \
  --num_drop_columns 5 \
  --continuous_feature_indices 0,1,2,3,4,5 \
  --seed 42 \
  --save_mask
```

### Output files
The tool creates:

- a modified `.csv`
- a companion metadata `.json`
- an optional sibling `feature_drop_mask` `.csv` when `--save_mask` is used

In `drop` and `both` modes, the output CSV has fewer covariate columns than the input because the selected features are removed rather than overwritten.

### Reproducibility
`--seed` controls every random decision in the pipeline, including dropped column selection and Gaussian draws. Reusing the same input, configuration, and seed gives the same result.

### Notes and limitations
- The current implementation uses only NumPy and the Python standard library.
- The perturbations target covariates only and do not simulate missingness mechanisms beyond structured feature drop.
- Gaussian noise is applied to all covariates unless `--continuous_feature_indices` is supplied explicitly.
- Natural future extensions include MAR or MNAR missingness, hidden confounding, or noisy outcomes.

### Quick sanity checks
You can verify the generated file with a short Python check:

```python
import json
import numpy as np

original = np.loadtxt("ihdp_npci_1.csv", delimiter=",")
modified = np.loadtxt("ihdp_both.csv", delimiter=",")
metadata = json.load(open("ihdp_both.json", "r", encoding="utf-8"))

assert np.array_equal(original[:, :5], modified[:, :5])
assert modified.shape[1] == original.shape[1] - len(metadata["dropped_column_indices"])

drop_mask = np.loadtxt("ihdp_both_feature_drop_mask.csv", delimiter=",")
dropped_columns = metadata["dropped_column_indices"]
keep_columns = [i for i in range(original.shape[1] - 5) if i not in dropped_columns]

# In drop mode, modified[:, 5:] should equal original[:, 5 + keep_columns].
# In combined mode, the same remaining columns are present, but Gaussian
# noise changes their values.
```
