from Forest.forest import PROJECT_ROOT
from noisify_ihdp import apply_gaussian_noise
import numpy as np

path = PROJECT_ROOT / "data" / "ihdp_dataset" / "csv" / "ihdp_npci_1.csv"
noisy_root = PROJECT_ROOT / "experiments" / "knn_counterfactual" / "noisy"

data = np.loadtxt(path, delimiter=",")
x = data[:, 5:]

out_dir = noisy_root / "gaussianSTD_test"
out_dir.mkdir(parents=True, exist_ok=True)


def pretty_float(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    return np.format_float_positional(value, trim="-")

for i in range(5):
    std = i * 0.2
    std_label = f"{std:.1f}"
    data_noisy = data.copy()
    data_noisy[:, 5:] = apply_gaussian_noise(
        x,
        gaussian_std=std,
        gaussian_mean=0.0,
        rng=np.random.default_rng(42),
    )
    output_path = out_dir / f"ihdp_npci_1_noisy_std_{std_label}.csv"
    with output_path.open("w", newline="") as handle:
        for row in data_noisy:
            handle.write(",".join(pretty_float(value) for value in row) + "\n")
