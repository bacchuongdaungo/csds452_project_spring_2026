from Forest.forest import PROJECT_ROOT
from noisify_ihdp import apply_gaussian_noise
import numpy as np
import pandas as pd


path = PROJECT_ROOT / "data" / "ihdp_dataset" / "csv" / "ihdp_npci_1.csv"
noisy_root = PROJECT_ROOT / "experiments" / "knn_counterfactual" / "noisy"

data = np.loadtxt(path, delimiter=",")
x = data[:, 5:]

base_columns = [
    "treatment",
    "y_factual",
    "y_cfactual",
    "mu0",
    "mu1",
]
feature_columns = [f"x{i}" for i in range(1, x.shape[1] + 1)]
columns = [*base_columns, *feature_columns]

noisy_root.mkdir(parents=True, exist_ok=True)


def pretty_float(value: float) -> str:
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    return np.format_float_positional(value, trim="-")


def write_csv(data_array: np.ndarray, output_path, csv_columns=None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data_array, columns=csv_columns or columns)
    df = df.map(pretty_float)
    df.to_csv(output_path, index=False)

'''
for i in range(5):
    std = (i + 1) * 0.2
    std_label = f"{std:.1f}"
    data_noisy = data.copy()
    data_noisy[:, 5:] = apply_gaussian_noise(
        x,
        gaussian_std=std,
        gaussian_mean=0.0,
        rng=np.random.default_rng(42),
    )
    output_path = noisy_root / f"ihdp_npci_1_noisy_std_{std_label}.csv"
    write_csv(data_noisy, output_path)
'''
'''
drop_rng = np.random.default_rng(42)
all_feature_indices = np.arange(x.shape[1])
dropped_columns: list[int] = []

for _ in range(5):
    remaining_columns = np.setdiff1d(
        all_feature_indices,
        np.array(dropped_columns, dtype=int),
        assume_unique=False,
    )
    new_columns = np.sort(drop_rng.choice(remaining_columns, size=3, replace=False)).tolist()
    dropped_columns.extend(new_columns)
    total_drop_columns = len(dropped_columns)
    keep_x_columns = np.setdiff1d(
        all_feature_indices,
        np.array(dropped_columns, dtype=int),
        assume_unique=False,
    )
    reduced_data = np.concatenate(
        [data[:, :5], data[:, 5:][:, keep_x_columns]],
        axis=1,
    )
    reduced_columns = [*base_columns, *[feature_columns[index] for index in keep_x_columns]]
    output_path = noisy_root / "drop_3_rep" / f"ihdp_npci_1_noisy_drop_{total_drop_columns}.csv"
    write_csv(reduced_data, output_path, csv_columns=reduced_columns)
    '''
#Both noise and drop
data_noisy = data.copy()
data_noisy[:, 5:] = apply_gaussian_noise(
        x,
        gaussian_std=0.5,
        gaussian_mean=0.0,
        rng=np.random.default_rng(42),
    )
drop_rng = np.random.default_rng(42)
all_feature_indices = np.arange(x.shape[1])
dropped_columns: list[int] = []

for _ in range(5):
    remaining_columns = np.setdiff1d(
        all_feature_indices,
        np.array(dropped_columns, dtype=int),
        assume_unique=False,
    )
    new_columns = np.sort(drop_rng.choice(remaining_columns, size=3, replace=False)).tolist()
    dropped_columns.extend(new_columns)
    total_drop_columns = len(dropped_columns)
    keep_x_columns = np.setdiff1d(
        all_feature_indices,
        np.array(dropped_columns, dtype=int),
        assume_unique=False,
    )
    reduced_data = np.concatenate(
        [data[:, :5], data[:, 5:][:, keep_x_columns]],
        axis=1,
    )
    reduced_columns = [*base_columns, *[feature_columns[index] for index in keep_x_columns]]
    output_path = noisy_root / "both_noise" / f"ihdp_npci_1_noisy_drop_{total_drop_columns}.csv"
    write_csv(reduced_data, output_path, csv_columns=reduced_columns)